import json
import sys


def extract_case_ids(patient_file, json_file, output_file):
    # Read the patient IDs from the text file
    with open(patient_file, "r") as f:
        patient_ids = set(f.read().strip().split(","))

    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Extract case IDs corresponding to the patient IDs
    case_ids = []
    for patient_id, cases in data.items():
        if patient_id in patient_ids:
            case_ids.extend(cases.keys())

    # Write the case IDs to the output file
    with open(output_file, "w") as f:
        f.write(",".join(case_ids))


if __name__ == "__main__":
    # Ensure three arguments are provided: patient file, json file, and output file
    if len(sys.argv) != 4:
        print("Usage: python script.py <patient_file> <json_file> <output_file>")
        sys.exit(1)

    # Assign arguments to variables
    patient_file = sys.argv[1]
    json_file = sys.argv[2]
    output_file = sys.argv[3]

    # Run the extraction
    extract_case_ids(patient_file, json_file, output_file)
