import os
import json
import random
from sklearn.model_selection import train_test_split


def create_train_val_test_splits(
    root_dir="data/hipt_superbatches",
    reports_file="data/reports/dummy_reports.json",
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    seed=42,
):
    """
    Creates train, validation, and test splits from the synthetic dataset and saves them to separate directories.

    Args:
        root_dir: Root directory containing the 'hipt_superbatches' data.
        reports_file: Path to the 'dummy_reports.json' file containing text data.
        train_size: Proportion of data for the training set.
        val_size: Proportion of data for the validation set.
        test_size: Proportion of data for the test set.
        seed: Random seed for reproducibility.
    """

    # Load text data from reports file
    with open(reports_file, "r") as f:
        text_data = json.load(f)

    # Get all patient IDs from the text data
    patient_ids = list(text_data.keys())

    # Set random seed for reproducibility
    random.seed(seed)

    # Split patient IDs into train, validation, and test sets
    train_ids, temp_ids = train_test_split(
        patient_ids, train_size=train_size, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_size / (val_size + test_size), random_state=seed
    )

    # Create output directories
    output_dirs = ["data/train_split", "data/val_split", "data/test_split"]
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(os.path.join(dir_name, "extracted_features"), exist_ok=True)

    # Copy relevant data to each split directory
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split_name, patient_ids in splits.items():
        output_dir = os.path.join("data", split_name)

        # # Copy feature files
        # for superbatch_dir in os.listdir(root_dir):
        #     extracted_features_dir = os.path.join(root_dir, superbatch_dir, "extracted_features")
        #     for file_name in os.listdir(extracted_features_dir):
        #         if file_name.endswith(".pth"):
        #             specimen_index = int(file_name.split(".")[0])
        #             patient_id = f"P{specimen_index:06d}"
        #             if patient_id in patient_ids:
        #                 shutil.copy(
        #                     os.path.join(extracted_features_dir, file_name),
        #                     os.path.join(output_dir, "extracted_features", file_name)
        #                 )

        # Copy and filter text data
        split_text_data = {
            patient_id: text_data[patient_id] for patient_id in patient_ids
        }
        with open(os.path.join(output_dir, "reports.json"), "w") as f:
            json.dump(split_text_data, f, indent=4)

        # Save patient IDs to a file
        with open(os.path.join(output_dir, "patient_ids.txt"), "w") as f:
            f.write(",".join(patient_ids))


if __name__ == "__main__":
    create_train_val_test_splits()
