import pandas as pd
import json


def create_specimen_json(excel_path, class_output_path, label_output_path):
    # Read the first sheet of the Excel file
    df = pd.read_excel(excel_path, sheet_name=0)

    # Ensure the required columns exist
    required_columns = [
        "patient",
        "specimen",
        "diagnosis",
        "diagnosis code",
        "secondary_findings",
        "label",
        "sex",
        "age",
        "location",
        "set",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the Excel file.")

    # Create dictionaries for mappings
    specimen_to_class_map = {}
    specimen_to_label_map = {}

    # Define label mapping: TRUE -> 1, FALSE -> 2
    label_mapping = {"TRUE": 1, "FALSE": 2}

    for _, row in df.iterrows():
        specimen = str(row["specimen"])

        # Cast diagnosis_code to int
        diagnosis_code = int(row["diagnosis code"])

        # Map label values (assuming they are strings "TRUE" or "FALSE")
        raw_label = str(row["label"]).upper()  # Ensure uppercase for a stable mapping
        if raw_label not in label_mapping:
            raise ValueError(
                f"Unexpected label value '{raw_label}' found for specimen {specimen}. Expected 'TRUE' or 'FALSE'."
            )
        label = label_mapping[raw_label]

        # Map each specimen to its diagnosis code (integer)
        specimen_to_class_map[specimen] = diagnosis_code

        # Map each specimen to its label (1 or 2)
        specimen_to_label_map[specimen] = label

    # Write the specimen_to_class_map to a JSON file
    with open(class_output_path, "w", encoding="utf-8") as f:
        json.dump(specimen_to_class_map, f, indent=4, ensure_ascii=False)
    print(f"Specimen to class map created at: {class_output_path}")

    # Write the specimen_to_label_map to a JSON file
    with open(label_output_path, "w", encoding="utf-8") as f:
        json.dump(specimen_to_label_map, f, indent=4, ensure_ascii=False)
    print(f"Specimen to label map created at: {label_output_path}")


if __name__ == "__main__":
    excel_path = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels/patient_characteristics.xlsx"
    class_output_path = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels/specimen_to_class_map.json"
    label_output_path = "/Users/sander.moonemans/Study/MasterThesisRepo/data/labels/specimen_to_label_map.json"
    create_specimen_json(excel_path, class_output_path, label_output_path)
