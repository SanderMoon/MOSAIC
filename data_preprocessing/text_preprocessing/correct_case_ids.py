import json
import os

files = ["data/reports/HE_HE_IHC_CON_subset.json", "data/reports/HE_HE_IHC_set.json"]

for file in files:
    with open(file, "r") as f:
        data = json.load(f)

    for patient_id in list(data.keys()):
        case_ids = list(data[patient_id].keys())
        new_data = {}

        for case_id in case_ids:
            new_case_id = case_id.replace("_", "")
            # Check for existing keys to avoid overwriting
            if new_case_id in new_data:
                raise ValueError(
                    f"Key collision detected: {new_case_id} already exists for patient {patient_id}"
                )
            new_data[new_case_id] = data[patient_id].pop(case_id)

        # Replace the patient's case data with the modified version
        data[patient_id] = new_data

    # Save to a new file with '_corrected' before the .json extension
    new_file_name = f"{os.path.splitext(file)[0]}_corrected.json"
    with open(new_file_name, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
