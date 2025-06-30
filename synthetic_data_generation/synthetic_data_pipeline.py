import os
import csv
import random
from PIL import Image
from synthetic_data_generator import SyntheticDataGenerator
from image_embedder import ImageEmbedder
import pandas as pd
import numpy as np
import torch
import h5py
from typing import List, Tuple
from config import ExperimentConfig  # Ensure correct import path
from sklearn.model_selection import train_test_split
import json


class SyntheticDataPipeline:
    """
    A pipeline to generate synthetic image-text data, embed images, and save them to a specified directory.
    Additionally, it structures the data into an HDF5 file following a predefined hierarchy.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initializes the pipeline with generator, dataset, and embedding configurations.

        Parameters:
        - config: Instance of ExperimentConfig containing all necessary configurations.
        """
        self.config = config
        self.output_dir = self.config.output_dir
        self.class_configs = self.config.class_configs
        self.embedder_config = self.config.embedder_config
        self.dataset_config = self.config.dataset_config
        self.split_config = self.config.split_config
        self.num_tiles_base = self.config.num_tiles_base
        self.image_size = (
            self.config.image_size[0] * self.num_tiles_base,
            self.config.image_size[1] * self.num_tiles_base,
        )

        # Initialize the SyntheticDataGenerator with generator parameters
        self.generator = SyntheticDataGenerator(
            image_size=self.image_size,  # Dynamic image size from config
            noise_level=self._get_average_noise_level(),
            base_image_size=self.image_size,  # Set base_image_size to match image_size for scaling
        )

        # Initialize the embedder if embedder configuration is provided
        if self.embedder_config:
            self.embedder = ImageEmbedder(config=self.embedder_config)
        else:
            self.embedder = None

        # Setup directory structure
        self.setup_directory()

        # Initialize a list to hold annotations
        self.annotations = []

    def _get_average_noise_level(self):
        """
        Calculates the average noise level from all class configurations that have image augmentation enabled.

        Returns:
        - Float representing the average noise level.
        """
        noise_levels = [
            config.augment_images_noise_level
            for config in self.class_configs
            if config.augment_images
        ]
        return np.mean(noise_levels) if noise_levels else self.generator.noise_level

    def setup_directory(self):
        """
        Sets up the output directory and subdirectories for images and embeddings.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        else:
            print(f"Directory already exists: {self.output_dir}")

        # Create subdirectories for images
        self.images_dir = os.path.join(self.output_dir, "images")
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            print(f"Created images directory: {self.images_dir}")
        else:
            print(f"Images directory already exists: {self.images_dir}")

        # Create subdirectories for visualizations if embedder is enabled
        if self.embedder:
            self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
            if not os.path.exists(self.visualizations_dir):
                os.makedirs(self.visualizations_dir)
                print(f"Created visualizations directory: {self.visualizations_dir}")
            else:
                print(
                    f"Visualizations directory already exists: {self.visualizations_dir}"
                )

            # Path to save embeddings
            self.embeddings_path = os.path.join(
                self.output_dir,
                self.dataset_config.hdf5_filename.replace(".hdf5", "_embeddings.csv"),
            )

    def generate_and_save(self):
        """
        Generates synthetic samples and saves images and their annotations.
        """
        embedder_size = (
            self.embedder.expected_size if self.embedder else self.generator.image_size
        )
        tile_base = self.num_tiles_base
        tile_size = embedder_size  # Each tile should match the embedder's expected size

        for class_config in self.class_configs:
            patient_id = class_config.name  # Assuming 'name' is the unique patient_id
            for sample_idx in range(1, class_config.num_samples + 1):
                case_id = f"Case{sample_idx}"  # Assigning Case1, Case2, ..., CaseN

                if tile_base == 1:
                    # Single-tile image generation
                    image, text = self.generator.generate_sample(class_config)
                    # Define a unique filename based on patient_id and case_id
                    filename = self._generate_filename(patient_id, case_id)
                    # Save the image
                    image_path = os.path.join(self.images_dir, filename)
                    image.save(image_path)
                    # Append annotation without tile coordinates
                    self.annotations.append(
                        {
                            "filename": filename,
                            "description": text,
                            "label": f"{patient_id}_{case_id}",
                            "tile_x": None,
                            "tile_y": None,
                        }
                    )
                else:
                    # Multi-tile image generation
                    large_image_size = (
                        tile_size[0] * tile_base,
                        tile_size[1] * tile_base,
                    )
                    # Generate the large image
                    image_size_tuple = (large_image_size[0], large_image_size[1])
                    large_image, text = self.generator.generate_sample(
                        class_config, image_size=image_size_tuple
                    )
                    # Define a unique filename for the large image
                    large_filename = self._generate_filename(
                        patient_id, case_id, is_large=True
                    )
                    # Save the large image
                    large_image_path = os.path.join(self.images_dir, large_filename)
                    large_image.save(large_image_path)
                    print(f"Saved large image: {large_image_path}")

                    # Split the large image into tiles
                    tiles = self._split_into_tiles(large_image, tile_base, tile_size)

                    for x in range(tile_base):
                        for y in range(tile_base):
                            tile_image = tiles[x][y]
                            # Define a unique filename for the tile
                            tile_case_id = (
                                f"{case_id}_Tile{x+1}_{y+1}"  # e.g., Case1_Tile1_1
                            )
                            tile_filename = self._generate_filename(
                                patient_id, tile_case_id, is_large=False
                            )
                            # Save the tile image
                            tile_image_path = os.path.join(
                                self.images_dir, tile_filename
                            )
                            tile_image.save(tile_image_path)
                            print(f"Saved tile image: {tile_image_path}")

                            # Append annotation with tile coordinates
                            self.annotations.append(
                                {
                                    "filename": tile_filename,
                                    "description": text,
                                    "label": f"{patient_id}_{case_id}",
                                    "tile_x": x + 1,  # 1-based indexing
                                    "tile_y": y + 1,
                                }
                            )

    def _split_into_tiles(
        self, large_image: Image.Image, tile_base: int, tile_size: Tuple[int, int]
    ) -> List[List[Image.Image]]:
        """
        Splits a large image into smaller tiles based on the tile base.

        Parameters:
        - large_image: PIL Image object of the large image.
        - tile_base: Number of tiles per row and column (e.g., 3 for 3x3 grid).
        - tile_size: Tuple specifying the size of each tile (width, height).

        Returns:
        - 2D list of PIL Image objects representing the tiles.
        """
        tiles = []
        for x in range(tile_base):
            row = []
            for y in range(tile_base):
                left = y * tile_size[0]
                upper = x * tile_size[1]
                right = left + tile_size[0]
                lower = upper + tile_size[1]
                tile = large_image.crop((left, upper, right, lower))
                row.append(tile)
            tiles.append(row)
        return tiles

    def _generate_filename(
        self, patient_id: str, case_id: str, is_large: bool = False
    ) -> str:
        """
        Generates a unique filename for each image or tile.

        Parameters:
        - patient_id: Unique identifier for the patient.
        - case_id: Unique identifier for the case or tile.
        - is_large: Boolean indicating whether the filename is for a large image.

        Returns:
        - String filename.
        """
        if is_large:
            base = f"{patient_id}_{case_id}_large_{random.randint(1000,9999)}.png"
        else:
            base = f"{patient_id}_{case_id}.png"

        # Ensure filename uniqueness
        while os.path.exists(os.path.join(self.images_dir, base)):
            if is_large:
                base = f"{patient_id}_{case_id}_large_{random.randint(1000,9999)}.png"
            else:
                base = f"{patient_id}_{case_id}_{random.randint(1000,9999)}.png"

        return base

    def save_annotations(self):
        """
        Saves the annotations to a CSV file in the output directory.
        """
        if not self.annotations:
            print("No annotations to save.")
            return
        annotations_file = os.path.join(self.output_dir, "annotations.csv")
        keys = self.annotations[0].keys()
        with open(annotations_file, "w", newline="", encoding="utf-8") as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.annotations)
        print(f"Saved annotations to {annotations_file}")

    def embed_images(self):
        """
        Embeds all images using the ImageEmbedder and saves the embeddings.
        Also saves a visualization for the first image or tile.
        """
        if not self.embedder:
            print("No embedder initialized. Please provide embedder configuration.")
            return

        embeddings = []
        visualization_saved = False

        for idx, annotation in enumerate(self.annotations):
            image_path = os.path.join(self.images_dir, annotation["filename"])
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                continue

            # Generate embedding
            embedding = self.embedder.embed_image(image)
            embeddings.append(
                {"filename": annotation["filename"], "embedding": embedding.tolist()}
            )

            # Save visualization for the first image or first tile
            if not visualization_saved:
                visualization_path = os.path.join(
                    self.visualizations_dir, "preprocessing_visualization.png"
                )
                self.embedder.save_preprocessed_visualization(image, visualization_path)
                visualization_saved = True

        if not embeddings:
            print("No embeddings were generated.")
            return

        # Save embeddings to a CSV file
        with open(self.embeddings_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            embedding_length = len(embeddings[0]["embedding"])
            header = ["filename"] + [f"embedding_{i}" for i in range(embedding_length)]
            writer.writerow(header)
            # Write data
            for embed in embeddings:
                writer.writerow([embed["filename"]] + embed["embedding"])

        print(f"Saved embeddings to {self.embeddings_path}")

    def run(self):
        """
        Runs the entire pipeline: generate data and save annotations.
        """
        self.generate_and_save()
        self.save_annotations()
        print("Data generation pipeline completed.")

    def run_with_embedding(self):
        """
        Runs the entire pipeline including embedding images.
        """
        self.run()
        self.embed_images()
        print("Data generation and embedding pipeline completed.")

    def save_to_hdf5(self, hdf5_path: str):
        """
        Reads embeddings and text annotations and saves them into an HDF5 file with the specified structure.

        Parameters:
        - hdf5_path: Path to the HDF5 file to be created.
        """
        if not self.embedder:
            print("No embedder initialized. Please provide embedder configuration.")
            return

        annotations_file = os.path.join(self.output_dir, "annotations.csv")
        embeddings_file = self.embeddings_path  # Already saved during embedding

        # Read annotations and embeddings
        try:
            annotations_df = pd.read_csv(annotations_file)
            embeddings_df = pd.read_csv(embeddings_file)
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            return

        # Merge on 'filename'
        data_df = pd.merge(annotations_df, embeddings_df, on="filename")

        if data_df.empty:
            print("Merged DataFrame is empty. Nothing to save to HDF5.")
            return

        # Parse patient ID and case ID from label
        # Assuming label format: PXXXXXX_CaseY or PXXXXXX_CaseY_TileX_Y
        def parse_label(label):
            parts = label.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Label '{label}' does not match the expected format 'PXXXXXX_CaseY'"
                )
            patient_id = parts[0]
            case_id = "_".join(parts[1:])  # In case case_id contains underscores
            return patient_id, case_id

        # Prepare lists for case-index
        case_ids_list = []  # For HDF5 (bytes)
        case_ids_txt_list = []  # For txt file (strings)
        case_paths_list = []

        # Open HDF5 file
        with h5py.File(hdf5_path, "w") as hdf5_file:
            # Create 'case-index' group
            case_index_grp = hdf5_file.create_group("case-index")

            # Create 'patients' group
            patients_grp = hdf5_file.create_group("patients")

            # Group data by label (patient_id and case_id)
            grouped = data_df.groupby("label")

            for label, group in grouped:
                try:
                    patient_id, case_id = parse_label(label)
                except ValueError as ve:
                    print(f"Skipping label '{label}': {ve}")
                    continue

                # Define HDF5 paths
                case_grp_path = f"patients/{patient_id}/{case_id}"

                # Create patient group if it doesn't exist
                if patient_id not in patients_grp:
                    patient_grp = patients_grp.create_group(patient_id)
                else:
                    patient_grp = patients_grp[patient_id]

                # Create case group if it doesn't exist
                if case_id not in patient_grp:
                    case_grp = patient_grp.create_group(case_id)
                else:
                    case_grp = patient_grp[case_id]

                # Create hipt_features_comp_1 group if it doesn't exist
                if "hipt_features_comp_1" not in case_grp:
                    hipt_features_grp = case_grp.create_group("hipt_features_comp_1")
                else:
                    hipt_features_grp = case_grp["hipt_features_comp_1"]

                # Extract embeddings
                embedding_cols = [
                    col for col in embeddings_df.columns if col.startswith("embedding_")
                ]
                embeddings = group[embedding_cols].values.astype(
                    np.float64
                )  # Shape: (n, embedding_dim)
                n_samples = embeddings.shape[0]
                embedding_dim = embeddings.shape[1]

                #
                embeddings_reshaped = embeddings.reshape(
                    n_samples, 1, embedding_dim
                )  # Shape: (n, D)

                # Generate positions based on tile_x and tile_y
                if self.num_tiles_base == 1:
                    # For single-tile images, assign a default position
                    positions_reshaped = np.array(
                        [[1, 1, 1] for _ in range(n_samples)], dtype="int64"
                    )  # [n, 3]
                else:
                    # For multi-tile images, use tile_x and tile_y
                    positions = group[["tile_x", "tile_y"]].values  # Shape: (n, 2)
                    positions_reshaped = np.hstack(
                        (
                            np.ones(
                                (n_samples, 1), dtype="int64"
                            ),  # Adding the first dimension as 1
                            positions,
                        )
                    )  # Shape: (n, 3)

                # Reshape positions to (n, 1, 3) as per requirement
                positions_reshaped = positions_reshaped.reshape(n_samples, 1, 3)

                # Generate tile_keys as unique identifiers per tile
                tile_keys = np.array(
                    [f"tile_{random.randint(1000,9999)}" for _ in range(n_samples)],
                    dtype="S13",
                )  # Shape: (n,)

                # Store datasets
                hipt_features_grp.create_dataset(
                    "features", data=embeddings_reshaped, dtype="float64"
                )
                hipt_features_grp.create_dataset(
                    "positions", data=positions_reshaped, dtype="int64"
                )
                hipt_features_grp.create_dataset(
                    "tile_keys", data=tile_keys, dtype="S13"
                )

                # Concatenate all descriptions for the case
                descriptions = group["description"].tolist()
                concatenated_text = descriptions[0]
                text_key = self.dataset_config.text_key
                dt = h5py.string_dtype(encoding="utf-8")
                case_grp.create_dataset(text_key, data=concatenated_text, dtype=dt)

                # Update case-index
                case_ids_list.append(
                    case_id.encode("utf-8")
                )  # HDF5 expects bytes for |S dtype
                case_paths_list.append(case_grp_path.encode("utf-8"))
                case_ids_txt_list.append(case_id)  # Store as string for txt file

            # After processing all cases, create datasets under 'case-index'
            if case_ids_list and case_paths_list:
                # Determine maximum length for dynamic string dtype
                max_case_id_length = max(len(cid) for cid in case_ids_list)
                max_case_path_length = max(len(cpath) for cpath in case_paths_list)

                case_ids_np = np.array(
                    case_ids_list, dtype=f"S{max_case_id_length}"
                )  # |S{max_length}
                case_paths_np = np.array(
                    case_paths_list, dtype=f"S{max_case_path_length}"
                )  # |S{max_length}

                # Create datasets
                case_index_grp.create_dataset("case_ids", data=case_ids_np)
                case_index_grp.create_dataset("case_paths", data=case_paths_np)

                print(f"Created 'case-index' with {len(case_ids_list)} cases.")
            else:
                print("No cases were added to 'case-index'.")

        train_ratio = self.config.split_config.train
        val_ratio = self.config.split_config.val
        test_ratio = self.config.split_config.test

        # Save caseIds to txt file comma separated without quotes
        if case_ids_txt_list:
            # Splitting the data into train and temp set
            train_ids, temp_ids = train_test_split(
                case_ids_txt_list, test_size=(1 - train_ratio), random_state=42
            )

            # Splitting the temp set into validation and test sets
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=(test_ratio / (test_ratio + val_ratio)),
                random_state=42,
            )

            splits = {"train": train_ids, "val": val_ids, "test": test_ids}

            for split, ids in splits.items():
                case_ids_joined = ",".join(ids)
                case_ids_path = os.path.join(self.output_dir, f"case_ids_{split}.txt")
                with open(case_ids_path, "w") as f:
                    f.write(case_ids_joined)
                print(f"Saved {split} case IDs to '{case_ids_path}'.")
        else:
            print("No case IDs to save to text files.")

    def save_to_file_structure(self, output_dir: str):
        """
        Reads embeddings and text annotations and saves them into the specified file structure.
        Additionally, creates a JSON file mapping patient IDs to case IDs and their text annotations.

        Parameters:
        - output_dir: Directory where the data will be saved.
        """

        if not self.embedder:
            print("No embedder initialized. Please provide embedder configuration.")
            return

        annotations_file = os.path.join(self.output_dir, "annotations.csv")
        embeddings_file = self.embeddings_path  # Already saved during embedding

        # Read annotations and embeddings
        try:
            annotations_df = pd.read_csv(annotations_file)
            embeddings_df = pd.read_csv(embeddings_file)
        except Exception as e:
            print(f"Error reading CSV files: {e}")
            return

        # Merge on 'filename'
        data_df = pd.merge(annotations_df, embeddings_df, on="filename")

        if data_df.empty:
            print("Merged DataFrame is empty. Nothing to save.")
            return

        # Parse patient ID and case ID from label
        # Assuming label format: PXXXXXX_CaseY or PXXXXXX_CaseY_TileX_Y
        def parse_label(label):
            parts = label.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"Label '{label}' does not match the expected format 'PXXXXXX_CaseY'"
                )
            patient_id = parts[0]
            case_id = "_".join(parts[1:])  # In case case_id contains underscores
            return patient_id, case_id

        # Add 'patient_id' and 'case_id' columns
        data_df[["patient_id", "case_id"]] = data_df["label"].apply(
            lambda x: pd.Series(parse_label(x))
        )

        # **NEW**: Create a dictionary for text annotations
        text_annotations = {}

        # Group by 'patient_id' and 'case_id' to collect descriptions
        grouped_descriptions = (
            data_df.groupby(["patient_id", "case_id"])["description"]
            .first()
            .reset_index()
        )

        # Build the nested dictionary
        for idx, row in grouped_descriptions.iterrows():
            patient_id = row["patient_id"]
            case_id = row["case_id"]
            description = row["description"]

            if patient_id not in text_annotations:
                text_annotations[patient_id] = {}
            text_annotations[patient_id][case_id] = description

        # Assign 'specimen_id' as 'case_id' (could be any unique identifier per specimen)
        data_df["specimen_id"] = data_df["case_id"]

        # Assign unique 'specimen_index' per 'specimen_id'
        specimen_ids = data_df["specimen_id"].unique()
        specimen_index_map = {
            specimen_id: idx for idx, specimen_id in enumerate(specimen_ids)
        }
        data_df["specimen_index"] = data_df["specimen_id"].map(specimen_index_map)

        # Assign 'slide_index' and 'cross_section_index' as 0 (assuming one slide and cross-section per specimen)
        data_df["slide_index"] = 0
        data_df["cross_section_index"] = 0

        # Assign 'tile_index' within each specimen
        data_df["tile_index"] = data_df.groupby("specimen_id").cumcount()

        # Get embedding columns
        embedding_cols = [
            col for col in embeddings_df.columns if col.startswith("embedding_")
        ]

        # Ensure 'tile_x' and 'tile_y' are present
        if "tile_x" not in data_df.columns or "tile_y" not in data_df.columns:
            print("Tile positions 'tile_x' and 'tile_y' are missing from data.")
            return

        # Now, process data in batches and save to file structure
        batch_size = 100  # Number of specimens per batch
        specimen_ids = data_df["specimen_id"].unique()
        num_batches = (len(specimen_ids) + batch_size - 1) // batch_size
        case_ids_txt_list = []  # For saving to txt file

        for batch_num in range(num_batches):
            batch_specimen_ids = specimen_ids[
                batch_num * batch_size: (batch_num + 1) * batch_size
            ]
            batch_dir = os.path.join(output_dir, f"data_{batch_num}")
            os.makedirs(batch_dir, exist_ok=True)
            extracted_features_dir = os.path.join(batch_dir, "extracted_features")
            os.makedirs(extracted_features_dir, exist_ok=True)

            # Paths to feature_information.txt and tile_information.txt
            feature_info_path = os.path.join(batch_dir, "feature_information.txt")
            tile_info_path = os.path.join(batch_dir, "tile_information.txt")

            with (
                open(feature_info_path, "w") as feature_info_file,
                open(tile_info_path, "w") as tile_info_file,
            ):
                for specimen_id in batch_specimen_ids:
                    case_ids_txt_list.append(
                        specimen_id
                    )  # Store as string for txt file
                    specimen_data = data_df[data_df["specimen_id"] == specimen_id]
                    specimen_index = specimen_data["specimen_index"].iloc[0]

                    # Get WSI names, assuming stored in 'filename' or 'original_wsi_name'
                    # For now, use 'filename' column
                    wsi_names = specimen_data["filename"].unique().tolist()
                    # create a string of wsi names like ['wsi1', 'wsi2', ...]
                    wsi_names = [f"{wsi_name}" for wsi_name in wsi_names]
                    # Write WSI names to feature_information.txt
                    feature_info_file.write(json.dumps(wsi_names) + "\n")

                    # Create specimen info
                    specimen_info = {
                        "specimen_index": int(specimen_index),
                        "patient": specimen_data["patient_id"].iloc[0],
                        "specimen": specimen_id,
                        "size": float(specimen_data.shape[0]),  # Number of tiles
                    }
                    # Write specimen info
                    feature_info_file.write(json.dumps(specimen_info) + "\n")

                    # Generate .pth filename
                    pth_filename_quoted = f'"{specimen_index}.pth"'
                    pth_filename = f"{specimen_index}.pth"

                    # Write pth_filename to feature_information.txt
                    feature_info_file.write(pth_filename_quoted + "\n")

                    # Now, create the .pth file
                    pth_file_path = os.path.join(extracted_features_dir, pth_filename)

                    # Build the data structure for the .pth file
                    pth_data = {0: {}}

                    # Now, for each row in specimen_data, create the entries
                    for idx, row in specimen_data.iterrows():
                        key_tuple = (
                            int(row["specimen_index"]),
                            int(row["slide_index"]),
                            int(row["cross_section_index"]),
                            int(row["tile_index"]),
                        )
                        # Feature is the embedding vector
                        feature_array = list(
                            row[embedding_cols].values.astype(np.float32)
                        )
                        feature_tensor = [feature_array]
                        # Position is torch.tensor([x, y, z])
                        # Assuming position is [1, tile_x, tile_y] as per previous code
                        position_tensor = [(1, int(row["tile_x"]), int(row["tile_y"]))]

                        # Since in the sample .pth file we have features under keys 0 and 1, with different sizes
                        # If you have different feature levels, you can adjust accordingly
                        pth_data[0][key_tuple] = {
                            "feature": feature_tensor,
                            "position": position_tensor,
                        }

                    # Save pth_data to .pth file
                    torch.save(pth_data, pth_file_path)

                    # Optionally, write to tile_information.txt if needed
                    # For demonstration, let's write the tile indices
                    tile_info = {
                        "specimen_index": int(specimen_index),
                        "tile_indices": specimen_data["tile_index"].tolist(),
                    }
                    tile_info_file.write(json.dumps(tile_info) + "\n")

            print(f"Saved data for batch {batch_num} to '{batch_dir}'.")

        # Save the text annotations dictionary to a JSON file
        text_annotations_path = os.path.join(self.output_dir, "text_annotations.json")
        with open(text_annotations_path, "w") as json_file:
            json.dump(text_annotations, json_file, indent=4)

        print(f"Saved text annotations to '{text_annotations_path}'.")

        train_ratio = self.config.split_config.train
        val_ratio = self.config.split_config.val
        test_ratio = self.config.split_config.test

        # Save caseIds to txt file comma separated without quotes
        if case_ids_txt_list:
            # Splitting the data into train and temp set
            train_ids, temp_ids = train_test_split(
                case_ids_txt_list, test_size=(1 - train_ratio), random_state=42
            )

            # Splitting the temp set into validation and test sets
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=(test_ratio / (test_ratio + val_ratio)),
                random_state=42,
            )

            splits = {"train": train_ids, "val": val_ids, "test": test_ids}

            for split, ids in splits.items():
                case_ids_joined = ",".join(ids)
                case_ids_path = os.path.join(self.output_dir, f"case_ids_{split}.txt")
                with open(case_ids_path, "w") as f:
                    f.write(case_ids_joined)
                print(f"Saved {split} case IDs to '{case_ids_path}'.")
        else:
            print("No case IDs to save to text files.")

    def create_zero_shot_labels(self):
        """
        Creates zero-shot learning label mappings in JSON format.

        There are two types of mappings:

        1. A mapping from textual descriptions of each class to a numeric class ID:
           Example:
           {
               "Histopathological evaluation of this skin specimen reveals a compound nevus.": 1,
               "Histopathological evaluation of this skin specimen reveals a dermal nevus.": 2,
               ...
           }

        2. A mapping from specimen ID (case ID) to class ID:
           Example:
           {
               "T13-00396I": 1,
               "T20-58255II": 2,
               ...
           }

        Parameters:

        Returns:
        None. Saves two JSON files in the output directory:
        - zero_shot_textual_to_id.json
        - zero_shot_specimen_to_id.json
        """

        # Create a class ID mapping for each class_config.
        # For simplicity, just enumerate them starting from 1.
        class_id_map = {}
        textual_to_id = {}

        # Assign a numeric ID for each class in the order they appear in class_configs
        for idx, class_config in enumerate(self.class_configs, start=1):
            # `class_config.name` should be something like "P000001_T20-58255II" or any unique ID
            # This will serve as specimen_id
            for i in range(1, class_config.num_samples + 1):
                specimen_id = f"{class_config.image_background_color}_Case{i}"
                class_id_map[specimen_id] = idx
                text_label = f"This image depicts a {class_config.image_background_color} background with a {class_config.image_bar_orientation} bar orientation and a {class_config.image_bar_thickness} bar thickness."
                textual_to_id[idx] = text_label

        # Save the mappings as JSON
        textual_to_id_path = os.path.join(
            self.output_dir, "zero_shot_textual_to_id.json"
        )
        specimen_to_id_path = os.path.join(
            self.output_dir, "zero_shot_specimen_to_id.json"
        )

        with open(textual_to_id_path, "w") as f:
            json.dump(textual_to_id, f, indent=4)

        with open(specimen_to_id_path, "w") as f:
            json.dump(class_id_map, f, indent=4)

        print(f"Saved zero-shot textual-to-ID labels to '{textual_to_id_path}'.")
        print(f"Saved zero-shot specimen-to-ID labels to '{specimen_to_id_path}'.")
