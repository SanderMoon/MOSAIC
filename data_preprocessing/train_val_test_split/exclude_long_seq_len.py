import h5py
import argparse
import sys
from transformers import AutoTokenizer
import warnings
from tqdm import tqdm
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create new train-val-test splits by excluding case IDs with image features or text sequences longer than specified thresholds."
    )
    parser.add_argument(
        "--hdf5_file", type=str, required=True, help="Path to the HDF5 file."
    )
    parser.add_argument(
        "--split_files",
        type=str,
        nargs=3,
        metavar=("train_split", "val_split", "test_split"),
        required=True,
        help="Paths to the train, validation, and test split txt files.",
    )
    parser.add_argument(
        "--text_attr",
        type=str,
        default="synthetic_text",
        help="Name of the text attribute to analyze (default: synthetic_text).",
    )
    parser.add_argument(
        "--max_features_seq_length",
        type=int,
        required=True,
        help="Maximum allowed sequence length for image features.",
    )
    parser.add_argument(
        "--max_text_seq_length",
        type=int,
        required=True,
        help="Maximum allowed sequence length for text.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_limited_seq",
        help="Suffix to append to the original split filenames for the new split files (default: _limited_seq).",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="excluded_cases.log",
        help="Path to the log file for excluded cases (default: excluded_cases.log).",
    )
    return parser.parse_args()


def find_datasets(file, target_names):
    """
    Efficiently traverse the HDF5 file to find all datasets matching target_names.

    :param file: Opened h5py File object.
    :param target_names: Set of dataset names to find.
    :return: Dictionary mapping target_names to lists of dataset paths.
    """
    datasets = {name: [] for name in target_names}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            base_name = name.split("/")[-1]
            if base_name in target_names:
                datasets[base_name].append(name)

    file.visititems(visitor)
    return datasets


def build_case_id_to_path_map(hdf5_file):
    """
    Build a mapping from case_id to case_path using the 'case-index' group in the HDF5 file.

    :param hdf5_file: Opened h5py File object.
    :return: Dictionary mapping case_id (str) to case_path (str).
    """
    case_index_group = hdf5_file.get("case-index")
    if not case_index_group:
        print("Error: 'case-index' group not found in the HDF5 file.")
        sys.exit(1)

    case_ids = case_index_group.get("case_ids")
    case_paths = case_index_group.get("case_paths")

    if case_ids is None or case_paths is None:
        print(
            "Error: 'case_ids' or 'case_paths' dataset not found in 'case-index' group."
        )
        sys.exit(1)

    case_id_list = [cid.decode("utf-8") for cid in case_ids[:]]
    case_path_list = [cpath.decode("utf-8") for cpath in case_paths[:]]

    if len(case_id_list) != len(case_path_list):
        print("Error: The number of case_ids and case_paths do not match.")
        sys.exit(1)

    case_id_to_path = dict(zip(case_id_list, case_path_list))
    return case_id_to_path


def read_split_file(split_file):
    """
    Read a split file and return a set of case_ids.

    :param split_file: Path to the split txt file.
    :return: Set of case_ids.
    """
    with open(split_file, "r") as f:
        content = f.read()
        case_ids = [cid.strip() for cid in content.split(",") if cid.strip()]
    return set(case_ids)


def write_split_file(original_split_file, new_split_file, included_case_ids):
    """
    Write the included case_ids to a new split file.

    :param original_split_file: Path to the original split file.
    :param new_split_file: Path to the new split file to be created.
    :param included_case_ids: Set of case_ids to include.
    """
    # Preserve the original order
    with open(original_split_file, "r") as f:
        content = f.read()
        all_case_ids = [cid.strip() for cid in content.split(",") if cid.strip()]

    included_ordered = [cid for cid in all_case_ids if cid in included_case_ids]

    with open(new_split_file, "w") as f:
        f.write(",".join(included_ordered))


def main():
    args = parse_arguments()
    hdf5_file_path = args.hdf5_file
    split_files = args.split_files  # [train_split, val_split, test_split]
    text_attr = args.text_attr
    max_features_seq_length = args.max_features_seq_length
    max_text_seq_length = args.max_text_seq_length
    output_suffix = args.output_suffix
    log_file = args.log_file

    # Initialize logging
    with open(log_file, "w") as log_f:
        log_f.write("Excluded Cases Log\n")
        log_f.write("===================\n\n")

    # Initialize the tokenizer
    try:
        print("Loading BioGPT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
    except Exception as e:
        print(f"Error loading BioGPT tokenizer: {e}")
        sys.exit(1)

    # Open the HDF5 file
    try:
        hdf5_file = h5py.File(hdf5_file_path, "r")
        print(f"Opened HDF5 file: {hdf5_file_path}")
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {hdf5_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening HDF5 file: {e}")
        sys.exit(1)

    # Build case_id to case_path mapping
    print("Building case ID to path mapping...")
    case_id_to_path = build_case_id_to_path_map(hdf5_file)
    print(f"Total case IDs in HDF5: {len(case_id_to_path)}")

    # Process each split file
    for split_file in split_files:
        split_name = os.path.splitext(os.path.basename(split_file))[
            0
        ]  # e.g., case_ids_train
        print(f"\nProcessing split: {split_name}")

        # Read case_ids from the split file
        split_case_ids = read_split_file(split_file)
        print(f"Original number of case IDs in {split_name}: {len(split_case_ids)}")

        included_case_ids = set()
        excluded_cases = []

        # Initialize progress bar
        print(f"Filtering cases in {split_name}...")
        for case_id in tqdm(
            split_case_ids, desc=f"Filtering {split_name}", unit="case"
        ):
            # Check if case_id exists in HDF5
            if case_id not in case_id_to_path:
                excluded_cases.append((case_id, "Case ID not found in HDF5"))
                continue

            case_path = case_id_to_path[case_id]

            # Define dataset paths
            features_ds_path = f"{case_path}/hipt_features_comp_1/features"
            text_ds_path = f"{case_path}/{text_attr}"

            # Initialize flags
            exclude = False
            reasons = []

            # Check features sequence length
            try:
                features_ds = hdf5_file[features_ds_path]
                features_length = features_ds.shape[0]
                if features_length > max_features_seq_length:
                    exclude = True
                    reasons.append(
                        f"Features seq length {features_length} > {max_features_seq_length}"
                    )
            except KeyError:
                exclude = True
                reasons.append("Features dataset not found")
            except Exception as e:
                exclude = True
                reasons.append(f"Error accessing features dataset: {e}")

            # Check text sequence length
            try:
                text_ds = hdf5_file[text_ds_path]
                # Ensure the dataset contains a string
                if text_ds.dtype.kind not in {"S", "O", "U"}:
                    exclude = True
                    reasons.append(
                        f"Text dataset dtype {text_ds.dtype} is not string-like"
                    )
                else:
                    # Read the string data
                    text = text_ds[()]
                    if isinstance(text, bytes):
                        text = text.decode("utf-8")
                    elif isinstance(text, str):
                        pass
                    else:
                        exclude = True
                        reasons.append(f"Unsupported text data type: {type(text)}")

                    if not exclude:
                        # Tokenize the text and count tokens
                        tokens = tokenizer.encode(text, add_special_tokens=True)
                        text_length = len(tokens)
                        if text_length > max_text_seq_length:
                            exclude = True
                            reasons.append(
                                f"Text seq length {text_length} > {max_text_seq_length}"
                            )
            except KeyError:
                exclude = True
                reasons.append("Text dataset not found")
            except Exception as e:
                exclude = True
                reasons.append(f"Error accessing text dataset: {e}")

            if not exclude:
                included_case_ids.add(case_id)
            else:
                excluded_cases.append((case_id, "; ".join(reasons)))

        # Write new split file
        new_split_file = split_file.replace(".txt", f"{output_suffix}.txt")
        write_split_file(split_file, new_split_file, included_case_ids)
        print(
            f"New split file created: {new_split_file} with {len(included_case_ids)} case IDs"
        )

        # Log excluded cases
        with open(log_file, "a") as log_f:
            log_f.write(f"Excluded Cases from {split_name}:\n")
            log_f.write("-----------------------------\n")
            for case_id, reason in excluded_cases:
                log_f.write(f"{case_id}: {reason}\n")
            log_f.write("\n")

        print(
            f"Excluded {len(excluded_cases)} cases from {split_name}. Details logged in {log_file}"
        )

    # Close the HDF5 file
    hdf5_file.close()
    print("\nProcessing complete.")


if __name__ == "__main__":
    # Suppress warnings from transformers library
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
