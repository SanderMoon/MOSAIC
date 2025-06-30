import os
import json


def create_specimen_index(root_dirs, output_file="specimen_to_patient_files.json"):
    specimen_to_info = {}

    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            if "feature_information.txt" in filenames:
                full_path_to_feature_info = os.path.join(
                    dirpath, "feature_information.txt"
                )
                with open(full_path_to_feature_info, "r") as f:
                    lines = f.readlines()
                    # Process lines in blocks of three
                    for i in range(0, len(lines), 3):
                        # Ensure we have at least three lines
                        if i + 2 < len(lines):
                            line2 = lines[i + 1].strip()
                            line3 = lines[i + 2].strip()
                            try:
                                specimen_info = json.loads(line2)
                                specimen_id = specimen_info["specimen"]
                                patient_id = specimen_info["patient"]
                                # Remove any enclosing quotation marks from the filename
                                feature_filename = line3.strip().strip('"').strip("'")
                                feature_file_path = os.path.join(
                                    dirpath, "extracted_features", feature_filename
                                )
                                # Add to mapping
                                if specimen_id not in specimen_to_info:
                                    specimen_to_info[specimen_id] = {
                                        "patient_id": patient_id,
                                        "files": [],
                                    }
                                specimen_to_info[specimen_id]["files"].append(
                                    os.path.abspath(feature_file_path)
                                )
                            except Exception as e:
                                print(
                                    f"Error processing lines {i}-{i+2} in file {full_path_to_feature_info}: {e}"
                                )
                                continue

    # Write the mapping to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(specimen_to_info, outfile, indent=4)
    print(f"Specimen index saved to {output_file}")


if __name__ == "__main__":
    root_dirs = ["hipt_superbatches", "hipt_superbatches_remainder"]
    create_specimen_index(root_dirs)
