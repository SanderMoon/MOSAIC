import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Value

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

"""
The expected directory structure for the HIPT dataset is as follows:
hipt_superbatches/
├─ data_0/
├─ data_1/
├─ data_2/
│ ├─ extracted_features/
│ │ ├─ 5.pth
│ │ ├─ 8.pth
│ ├─ feature_information.txt
│ ├─ tile_information.txt

Here the feature_information file is as follows:
["AAA-x-xxxxxx - YYYY-MM-DD hh.mm.ss.ndpi"]
{"specimen_index": x, "patient": "Pxxxxxx", "specimen": "Tx-xxxxxxA", "size": x.xx}
'x.pth'
["AAA-x-xxxxxx - YYYY-MM-DD hh.mm.ss.ndpi", "AAA-x-xxxxxx - YYYY-MM-DD hh.mm.ss.ndpi", "AAA-x-xxxxxx - YYYY-MM-DD hh.mm.ss.ndpi", "AAA-x-xxxxxx - YYYY-MM-DD hh.mm.ss.ndpi"]
{"specimen_index": x, "patient": "Pxxxxxx", "specimen": "Tx-xxxxxxA", "size": x.xx}
'x.pth'
...

With the first line containing the original whole slide image names, on the 2nd line the case information, and on the 3rd line the feature filename

Furthermore the .pth files have the following format:
{
    0: {
        (x, x, x ,x): {
            "feature": torch.tensor (N, 384),
            "position": torch.tensor (3)
        },
        (x, x, x ,y): {
            "feature": torch.tensor (N, 384),
            "position": torch.tensor (3)
        },
        ...
    },
    1: {
        (x, x, y ,y): {
            "feature": torch.tensor (1, 192),
            "position": torch.tensor (3)
        },
        (x, y, y ,y): {
            "feature": torch.tensor (1, 192),
            "position": torch.tensor (3)
        },
        ...
    }
}

Where the 0 are the features of the first HIPT component and the 1 are the features of the second HIPT component.
And the tuple key with 4 digits are (specimen index, slide index, cross-section index, tile index).
And position are the corresponding coordinates.

"""


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler | None = None
    shared_epoch: SharedEpoch | None = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class HDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file_path,
        component_name,
        text_attribute_name,
        case_id_file,
        tokenizer,
        transform=None,
        num_samples=None,
    ):
        """
        PyTorch Dataset to load data from the HDF5 file.

        Parameters:
            hdf5_file_path (str): Path to the HDF5 file.
            component_name (str): Name of the component (e.g., 'hipt_features_comp_1').
            text_attribute_name (str): Name of the text data attribute.
            tokenizer: Tokenizer for processing text data.
            transform: Optional transform to apply to the image features.
        """
        self.hdf5_file_path = hdf5_file_path
        self.component_name = component_name
        self.text_attribute_name = text_attribute_name
        self.transform = transform
        self.tokenizer = tokenizer
        self.case_id_file = case_id_file

        # Open the HDF5 file here
        self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

        # Build an index of available samples using the case_index
        self.sample_indices = self._build_index()

        if num_samples is not None:
            self.sample_indices = self.sample_indices[:num_samples]

        logging.info(
            f"Loaded {len(self.sample_indices)} samples from '{self.hdf5_file_path}'."
        )

    def __getstate__(self):
        # Exclude the HDF5 file handle from being pickled
        state = self.__dict__.copy()
        if "hdf5_file" in state:
            del state["hdf5_file"]
        return state

    def __setstate__(self, state):
        # Restore attributes and reopen the HDF5 file in the new process
        self.__dict__.update(state)
        self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

    def _build_index(self):
        """
        Build an index of samples using the case_index from the HDF5 file.
        """
        # Load patient IDs
        with open(self.case_id_file, "r") as f:
            case_ids_split = f.read().split(",")

        sample_indices = []
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            case_index_group = hdf5_file["case-index"]
            case_ids = case_index_group["case_ids"][:]
            case_paths = case_index_group["case_paths"][:]

            # Build a mapping from case IDs to paths
            case_index = {}
            for case_id, case_path in zip(case_ids, case_paths):
                case_id = case_id.decode("utf-8")
                case_path = case_path.decode("utf-8")
                case_index[case_id] = case_path

            # Extract patient IDs and specimen IDs from paths
            for case_id, case_path in case_index.items():
                # Paths are structured like '/patients/P001000/T21-76897I'
                path_parts = case_path.strip("/").split("/")
                if len(path_parts) == 3 and path_parts[0] == "patients":
                    patient_id = path_parts[1]
                    specimen_id = path_parts[2]
                    # Store indices as tuples
                    if case_id in case_ids_split:
                        sample_indices.append((patient_id, specimen_id, case_path))
                else:
                    print(f"Invalid case path: {case_path}")
        return sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            raise RuntimeError(
                "HDF5 file is not open. Please ensure the dataset is properly initialized."
            )
        # Get patient_id, specimen_id, and case_path
        patient_id, specimen_id, case_path = self.sample_indices[idx]

        # Navigate to the case group using the case_path
        case_group = self.hdf5_file[case_path]

        # Check if the component and text attribute exist
        if (
            self.component_name not in case_group
            or self.text_attribute_name not in case_group
        ):
            raise KeyError(
                f"Component '{self.component_name}' or text attribute '{self.text_attribute_name}' not found in case '{case_path}'."
            )

        # Load image features
        component_group = case_group[self.component_name]
        features = component_group["features"][:]
        positions = component_group["positions"][:]
        tile_keys = component_group["tile_keys"][:]

        # Apply transform if provided
        if self.transform:
            features = self.transform(features)
        else:
            # Convert to torch tensors
            features = torch.tensor(features, dtype=torch.float32)
            features = features.permute(1, 0, 2)
            positions = torch.tensor(positions, dtype=torch.float32)

        # Load and decode text data
        text_data = case_group[self.text_attribute_name][()]
        if isinstance(text_data, bytes):
            text_data = text_data.decode("utf-8")

        # Tokenize text
        text_inputs = self.tokenizer(
            text_data,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Add BOS and EOS tokens to input_ids
        text_inputs["input_ids"] = torch.cat(
            [
                torch.full(
                    (text_inputs["input_ids"].shape[0], 1), bos_id, dtype=torch.long
                ),
                text_inputs["input_ids"],
                torch.full(
                    (text_inputs["input_ids"].shape[0], 1), eos_id, dtype=torch.long
                ),
            ],
            dim=1,
        )

        # Update attention_mask
        text_inputs["attention_mask"] = torch.cat(
            [
                torch.ones(
                    (text_inputs["attention_mask"].shape[0], 1), dtype=torch.long
                ),
                text_inputs["attention_mask"],
                torch.ones(
                    (text_inputs["attention_mask"].shape[0], 1), dtype=torch.long
                ),
            ],
            dim=1,
        )

        return {
            "image_features": features,
            "positions": positions,
            "tile_keys": tile_keys,
            "text_inputs": text_inputs,
            "patient_id": patient_id,
            "specimen_id": specimen_id,
        }

    def __del__(self):
        if hasattr(self, "hdf5_file") and self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None

    def get_case_by_id(self, case_id):
        """
        Retrieves a sample from the dataset based on the given case ID.

        Args:
            case_id (str): The case ID to retrieve.

        Returns:
            dict: A dictionary containing the sample data if the case ID is found,
                  None otherwise.
        """
        # Find the index of the sample with the matching case_id
        idx = None
        for i, (patient_id, specimen_id, case_path) in enumerate(self.sample_indices):
            if specimen_id == case_id:
                idx = i
                break

        # If no sample is found with the given case_id
        if idx is None:
            logging.warning(f"Case ID '{case_id}' not found in the dataset.")
            return None

        # Use the existing __getitem__ method to retrieve the sample
        return self.__getitem__(idx)


class CaseDataset(Dataset):
    def __init__(
        self,
        case_id_file,
        text_data_file,
        root_dir,
        tokenizer,
        num_samples=None,
        comp_index=0,
    ):
        """
        Args:
            case_id_file (string): Path to the file with case_ids separated by commas.
            text_data_file (string): Path to the text annotations JSON file.
            root_dir (string): Root directory of the dataset containing the data batches.
            tokenizer: Tokenizer object used for text processing.
        """
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.comp_index = comp_index

        # Read case IDs
        with open(case_id_file, "r") as f:
            case_ids = f.read().strip().split(",")
        self.case_ids = [case_id.strip() for case_id in case_ids if case_id.strip()]
        self.case_ids = (
            self.case_ids[:num_samples] if num_samples is not None else self.case_ids
        )

        # Read text annotations
        with open(text_data_file, "r") as f:
            self.text_data = json.load(f)

        # Build index mapping from case IDs to .pth files
        self.case_index = {}

        # Iterate over all batch directories in root_dir and _remainder if present
        dirs = [self.root_dir, self.root_dir + "_remainder"]
        for root_dir in dirs:
            if not os.path.exists(root_dir):
                continue
            for batch_dir in os.listdir(root_dir):  # Use root_dir here
                batch_path = os.path.join(root_dir, batch_dir)  # And here
                if os.path.isdir(batch_path):
                    feature_info_path = os.path.join(
                        batch_path, "feature_information.txt"
                    )
                    extracted_features_dir = os.path.join(
                        batch_path, "extracted_features"
                    )
                    if os.path.exists(feature_info_path):
                        # Read feature_information.txt
                        with open(feature_info_path, "r") as f:
                            lines = f.readlines()
                        # Each entry consists of three lines:
                        # 1. WSI names (JSON array)
                        # 2. Specimen info (JSON object)
                        # 3. pth filename
                        i = 0
                        while i < len(lines):
                            specimen_info = json.loads(lines[i + 1].strip())
                            pth_filename = lines[i + 2].strip()
                            # remove quotes from pth_filename
                            pth_filename = pth_filename[1:-1]
                            i += 3
                            patient_id = specimen_info["patient"]
                            specimen_id = specimen_info["specimen"]

                            # Check if the specimen_id is in our case_ids
                            if specimen_id in self.case_ids:
                                pth_file_path = os.path.join(
                                    extracted_features_dir, pth_filename
                                )
                                self.case_index[specimen_id] = {
                                    "pth_file": pth_file_path,
                                    "patient_id": patient_id,
                                    "specimen_id": specimen_id,
                                    "batch_dir": batch_dir,
                                }
        # Update the list of case IDs to include only those found in the dataset
        logging.info(
            f"Moving on with {len(self.case_index)} case_ids vs case_id_file: {len(case_ids)}"
        )
        self.case_ids = list(self.case_index.keys())
        logging.info(
            f"Loaded {len(self.case_ids)} cases from '{case_id_file}' and '{root_dir}'."
        )

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # Get the case ID
        case_id = self.case_ids[idx]
        case_info = self.case_index[case_id]
        pth_file = case_info["pth_file"]
        patient_id = case_info["patient_id"]
        specimen_id = case_info["specimen_id"]

        # Load the .pth file
        try:
            pth_data = torch.load(pth_file)
        except Exception as e:
            raise RuntimeError(f"Error loading .pth file for case '{case_id}': {e}")

        # Access the data under key [self.comp_index]
        if self.comp_index not in pth_data:
            raise KeyError(f"Key '0' not found in .pth file for case '{case_id}'.")

        features_positions = pth_data[self.comp_index]

        # Extract features and positions
        features_list = []
        positions_list = []
        tile_keys = []
        for key_tuple, value in features_positions.items():
            # Convert features and positions to torch tensors with dtype float32
            feature = torch.tensor(value["feature"], dtype=torch.float32)
            position = torch.tensor(value["position"], dtype=torch.float32)
            features_list.append(feature)
            positions_list.append(position)
            tile_keys.append(key_tuple)

        if not features_list:
            raise ValueError(f"No features found for case '{case_id}'.")

        # Stack features and positions into tensors
        features = torch.stack(features_list)  # Shape: [num_tiles, feature_dim]
        positions = torch.stack(positions_list)  # Shape: [num_tiles, position_dim]

        # Convert tile_keys to a numpy array for consistency
        tile_keys = np.array(tile_keys)

        # Get the text data
        try:
            text_data = self.text_data[patient_id][case_id]
        except KeyError:
            raise KeyError(
                f"Text annotation not found for patient '{patient_id}', case '{case_id}'."
            )

        # Tokenize text
        text_inputs = self.tokenizer(
            text_data,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # Add BOS and EOS tokens to input_ids
        text_inputs["input_ids"] = torch.cat(
            [
                torch.full(
                    (text_inputs["input_ids"].shape[0], 1), bos_id, dtype=torch.long
                ),
                text_inputs["input_ids"],
                torch.full(
                    (text_inputs["input_ids"].shape[0], 1), eos_id, dtype=torch.long
                ),
            ],
            dim=1,
        )

        # Update attention_mask
        text_inputs["attention_mask"] = torch.cat(
            [
                torch.ones(
                    (text_inputs["attention_mask"].shape[0], 1), dtype=torch.long
                ),
                text_inputs["attention_mask"],
                torch.ones(
                    (text_inputs["attention_mask"].shape[0], 1), dtype=torch.long
                ),
            ],
            dim=1,
        )

        return {
            "image_features": features,
            "positions": positions,
            "tile_keys": tile_keys,
            "text_inputs": text_inputs,
            "patient_id": patient_id,
            "specimen_id": specimen_id,
        }

    def get_case_by_id(self, case_id):
        """
        Retrieves a sample from the dataset based on the given case ID.

        Args:
            case_id (str): The case ID to retrieve.

        Returns:
            dict: A dictionary containing the sample data if the case ID is found,
                  None otherwise.
        """
        if case_id not in self.case_ids:
            logging.warning(f"Case ID '{case_id}' not found in the dataset.")
            return None

        # Find the index of the case_id in the case_ids list
        idx = self.case_ids.index(case_id)

        # Use the existing __getitem__ method to retrieve the sample
        return self.__getitem__(idx)


def hdf5_collate_fn(
    batch, pad_token_id=1, image_feature_count_cutoff=100000
):  # 0 for first HIPT component, 1 for second
    """
    Collates a batch of data, pre-loading tensors, handling patient IDs, and applying padding.

    Args:
        batch: List of dictionaries from the dataset.
        pad_token_id: Token ID for padding text inputs.
        image_feature_count_cutoff: Maximum number of image features to consider.

    Returns:
        padded_features_batch: Padded batched tensor of features.
        padded_positions_batch: Padded batched tensor of positions.
        patient_ids_batch: List of patient IDs corresponding to the batch.
        text_inputs_batch: Batched text inputs.
    """

    image_features_list = []
    positions_list = []
    text_input_ids = []
    text_attention_masks = []
    patient_ids_batch = []
    specimen_ids_batch = []

    # Iterate over each sample in the batch
    for item in batch:
        # Extract and process image features
        C, N, D = item["image_features"].shape
        image_features = item["image_features"].reshape(C * N, D)  # Shape: [N, 192]
        # Random sample across image features if the count exceeds the cutoff
        if image_features.size(0) > image_feature_count_cutoff:
            indices = np.random.choice(
                image_features.size(0), image_feature_count_cutoff, replace=False
            )
            image_features = image_features[indices]
        image_features_list.append(image_features)

        # Extract and process positions
        positions = item["positions"]

        # Flatten the positions tensor if it's 3D (C, N, 3) -> (C*N, 3)
        if positions.dim() == 3:
            C_pos, N_pos, D_pos = positions.shape
            positions = positions.reshape(C_pos * N_pos, D_pos)

        positions_list.append(positions)

        # Collect patient and specimen IDs
        patient_ids_batch.append(item["patient_id"])
        specimen_ids_batch.append(item["specimen_id"])

        # Extract and process text inputs
        text_inputs = item["text_inputs"]
        input_ids = text_inputs["input_ids"].squeeze(0)  # Shape: [seq_len]
        attention_mask = text_inputs["attention_mask"].squeeze(0)  # Shape: [seq_len]
        text_input_ids.append(input_ids)
        text_attention_masks.append(attention_mask)

    # Pad image_features to the maximum N in the batch
    padded_image_features = pad_sequence(
        image_features_list, batch_first=True, padding_value=0
    )  # Shape: [batch_size, max_N, 192]

    # Pad positions to the maximum N in the batch
    try:
        padded_positions = pad_sequence(
            positions_list, batch_first=True, padding_value=0
        )  # Shape: [batch_size, max_N, P]
    except Exception as e:
        logging.exception(f"patient ids: {specimen_ids_batch}")
        logging.exception(f"shapes: {[position.shape for position in positions_list]}")
        raise e

    # Pad text input_ids and attention_masks to the maximum sequence length in the batch
    padded_text_input_ids = pad_sequence(
        text_input_ids, batch_first=True, padding_value=pad_token_id
    )  # Shape: [batch_size, max_seq_len]

    padded_text_attention_masks = pad_sequence(
        text_attention_masks, batch_first=True, padding_value=0
    )  # Shape: [batch_size, max_seq_len]

    feature_lengths = [features.shape[0] for features in image_features_list]
    batch_size = len(feature_lengths)
    max_len = padded_image_features.size(1)
    col_indices = torch.arange(max_len).unsqueeze(0).repeat(batch_size, 1)
    length_tensor = torch.tensor(feature_lengths).unsqueeze(1).repeat(1, max_len)

    image_feature_attention_mask = (col_indices < length_tensor).float()

    # Organize text inputs into a dictionary
    text_inputs_batch = {
        "input_ids": padded_text_input_ids,
        "attention_mask": padded_text_attention_masks,
    }

    # Organize image features into a dictionary with attention masks
    image_features_batch = {
        "features": padded_image_features,
        "attention_mask": image_feature_attention_mask,
    }

    return {
        "image_features": image_features_batch,  # Dict with 'features' and 'attention_mask'
        "positions": padded_positions,  # Tensor: [batch_size, max_N, P]
        "text_inputs": text_inputs_batch,  # Dict with 'input_ids' and 'attention_mask'
        "patient_ids": patient_ids_batch,  # List of patient IDs
        "specimen_ids": specimen_ids_batch,  # List of specimen IDs
    }


def get_dataset(split, is_train, tokenizer, args, num_training_samples=None):
    # hdf5_file_path= "../hipt_text_dataset.h5" args.hdf5_filename
    # component_name= "hipt_features_comp_1" args.hdf5_feature_attribute
    # text_attribute_name= "report_HE_IHC"    args.hdf5_text_attribute

    if args.hdf5_filename:
        dataset = HDF5Dataset(
            hdf5_file_path=args.hdf5_filename,
            component_name=args.hdf5_feature_attribute,
            text_attribute_name=args.hdf5_text_attribute,
            case_id_file=split,
            tokenizer=tokenizer,
            num_samples=num_training_samples if is_train else None,
        )
    else:
        root_dir = args.root_dir
        text_data_file = args.text_data_file
        if root_dir is None or text_data_file is None:
            raise ValueError(
                "root_dir and text_data_file must be provided for the CaseDataset."
            )
        dataset = CaseDataset(
            split,
            text_data_file=text_data_file,
            root_dir=args.root_dir,
            tokenizer=tokenizer,
            num_samples=num_training_samples if is_train else None,
        )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = is_train and sampler is None

    collate = hdf5_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=partial(
            collate,
            pad_token_id=tokenizer.pad_token_id,
            image_feature_count_cutoff=args.image_features_cutoff,
        ),
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(tokenizer, args, train_split=None, val_split=None, test_split=None):
    data = {}

    if train_split:
        data["train"] = get_dataset(
            train_split,
            is_train=True,
            tokenizer=tokenizer,
            args=args,
            num_training_samples=args.num_samples,
        )
    if val_split:
        data["val"] = get_dataset(
            val_split, is_train=False, tokenizer=tokenizer, args=args
        )
    if test_split:
        data["test"] = get_dataset(
            test_split, is_train=False, tokenizer=tokenizer, args=args
        )

    return data


if __name__ == "__main__":
    # import microsoft/biogpt
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")

    # Initialize the dataset
    dataset = CaseDataset(
        case_id_file="/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/case_ids_train.txt",
        text_data_file="/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/text_annotations.json",
        root_dir="/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/synth_superbatches",
        tokenizer=tokenizer,
    )

    # Access an item
    sample = dataset[0]

    print(sample)
