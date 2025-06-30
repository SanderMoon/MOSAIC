import json
import os


def remap_and_filter_json(input_file, output_file, label_map, keep_classes):
    """
    Remaps labels in a JSON file based on a provided mapping and filters
    entries based on a list of classes to keep.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
        label_map: A dictionary mapping original labels to new labels.
        keep_classes: A list of classes (after remapping) to keep in the output.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    remapped_data = {}
    for key, original_label in data.items():
        new_label = label_map.get(original_label)
        if new_label is not None and new_label in keep_classes:
            remapped_data[key] = new_label

    with open(output_file, "w") as f:
        json.dump(remapped_data, f, indent=4)


def main():
    input_file = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels/specimen_to_class_map.json"  # Replace with your actual input file
    output_dir = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels"  # Directory to store output files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Task 1: 4 Melanoma Subtypes ---
    melanoma_label_map = {
        4: 1,
        35: 1,
        40: 1,
        64: 1,  # Class 1: Superficial spreading melanoma
        5: 2,
        37: 2,  # Class 2: Nodular melanoma
        13: 3,
        54: 3,  # Class 3: Lentigo maligna melanoma
        15: 4,
        50: 4,  # Class 4: Acral/subungual melanoma
    }
    remap_and_filter_json(
        input_file,
        os.path.join(output_dir, "melanoma_subtypes.json"),
        melanoma_label_map,
        [1, 2, 3, 4],
    )

    # --- Task 2: 10 Melanocytic Lesion Subtypes ---
    ten_class_label_map = {
        **melanoma_label_map,  # Include melanoma subtypes
        7: 5,
        38: 5,  # Class 5: Blue nevus
        20: 6,  # Class 6: Halo nevus
        27: 7,
        57: 7,  # Class 7: Recurrent nevus
        8: 8,  # Class 8: Spitz nevus
        18: 9,
        39: 9,  # Class 9: BAP1-inactivated melanocytoma
    }
    remap_and_filter_json(
        input_file,
        os.path.join(output_dir, "ten_melanocytic_lesion_subtypes.json"),
        ten_class_label_map,
        list(range(1, 10)),  # Classes 1 to 9
    )

    # --- Task 3: Binary Melanoma vs. Melanocytic Nevi ---
    binary_melanoma_nevi_map = {
        **{k: 1 for k in melanoma_label_map},  # All melanoma subtypes to class 1
        **{
            k: 2 for k in ten_class_label_map if k not in melanoma_label_map
        },  # Nevi to class 2
    }
    remap_and_filter_json(
        input_file,
        os.path.join(output_dir, "binary_melanoma_vs_nevi.json"),
        binary_melanoma_nevi_map,
        [1, 2],
    )

    # --- Task 4: Malignant vs. Non-malignant ---
    malignant_nonmalignant_map = {
        **{k: 1 for k in melanoma_label_map},
        **{
            k: 2 for k in ten_class_label_map if k not in melanoma_label_map
        },  # Nevi to class 2
        2: 2,
        1: 2,
        3: 2,
        21: 2,  # Adding dermal, compound, junctional and multiple nevi.
    }

    remap_and_filter_json(
        input_file,
        os.path.join(output_dir, "malignant_nonmalignant.json"),
        malignant_nonmalignant_map,
        [1, 2],
    )

    # --- Task 5: Complex vs. Non-complex ---
    # here, common nevi are one class, and all other labels are the other class, we cannot use the previous mapping
    # as there are more classes in the total file than just the melanoma and nevi classes.
    non_complex_map = {2: 1, 1: 2, 3: 2, 21: 2}
    # From 1-65 there are 65 classes, so we can just add all the other classes to the non-complex class.
    complex_map = {k: 1 for k in range(1, 66) if k not in non_complex_map}

    complex_noncomplex_map = {**non_complex_map, **complex_map}
    remap_and_filter_json(
        input_file,
        os.path.join(output_dir, "complex_noncomplex.json"),
        complex_noncomplex_map,
        [1, 2],
    )


if __name__ == "__main__":
    main()
