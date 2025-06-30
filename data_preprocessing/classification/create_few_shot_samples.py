import random
import json


def get_and_store_x_shot_samples(
    training_ids_file, id_to_class_file, x, output_json_file
):
    """
    Extracts x samples per class from a training set, and stores the result in a JSON file.

    Args:
        training_ids_file: Path to the file containing training IDs (comma-separated).
        id_to_class_file: Path to the JSON file containing the ID to class mapping.
        x: The number of samples to extract per class (x-shot).
        output_json_file: Path to the JSON file where the x-shot samples will be stored.
    """

    # 1. Load training IDs
    with open(training_ids_file, "r") as f:
        training_ids_str = f.read().strip(",")
        training_ids = set(training_ids_str.split(","))

    # 2. Load ID to class mapping
    with open(id_to_class_file, "r") as f:
        id_to_class = json.load(f)

    # 3. Filter the mapping to include only IDs present in the training set
    training_id_to_class = {
        id: class_label for id, class_label in id_to_class.items() if id in training_ids
    }

    # 4. Organize IDs by class
    class_to_ids = {}
    for id, class_label in training_id_to_class.items():
        class_to_ids.setdefault(class_label, []).append(id)

    # 5. Extract x samples per class
    x_shot_samples = {}
    for class_label, ids in class_to_ids.items():
        if len(ids) >= x:
            x_shot_samples[str(class_label)] = random.sample(
                ids, x
            )  # Convert class label to string for JSON
        else:
            print(
                f"Warning: Class {class_label} has fewer than {x} samples in the training set. Using all available samples ({len(ids)})."
            )
            x_shot_samples[str(class_label)] = (
                ids  # Convert class label to string for JSON
            )

    # 6. Store the result in a JSON file
    with open(output_json_file, "w") as f:
        json.dump(x_shot_samples, f, indent=4)


# --- Example Usage ---
training_ids_file = "/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/case_ids_train.txt"  # Replace with your file path
id_to_class_file = "/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/zero_shot_specimen_to_id.json"  # Replace with your file path
x = 1
output_json_file = "/Users/sander.moonemans/Study/MasterThesisRepo/data/synthetic_data_multi_class_v2/x_shot_colors.json"  # Replace with your desired output file path

get_and_store_x_shot_samples(training_ids_file, id_to_class_file, x, output_json_file)

print(f"{x}-shot samples per class have been stored in {output_json_file}")
