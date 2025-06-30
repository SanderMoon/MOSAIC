#!/usr/bin/env python3

import json
import argparse
import sys
from collections import defaultdict


def load_json(file_path):
    """
    Load a JSON file and return its content as a dictionary.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        sys.exit(1)


def save_json(data, file_path):
    """
    Save a dictionary as a JSON file.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving JSON file '{file_path}': {e}")
        sys.exit(1)


def create_subset(file_primary, file_superset):
    """
    Create subsets based on the primary file from the superset file.

    Returns:
        subset_primary: subset of the primary data
        subset_superset: corresponding subset from the superset data
        missing_patients: list of patient IDs missing in the superset
        missing_cases: dict mapping patient IDs to lists of missing case IDs
    """
    subset_primary = {}
    subset_superset = {}
    missing_patients = []
    missing_cases = defaultdict(list)

    for patient_id, cases in file_primary.items():
        if patient_id not in file_superset:
            missing_patients.append(patient_id)
            continue  # Skip to next patient

        superset_cases = file_superset[patient_id]
        subset_primary[patient_id] = {}
        subset_superset[patient_id] = {}

        for case_id, text in cases.items():
            if case_id not in superset_cases:
                missing_cases[patient_id].append(case_id)
                continue  # Skip to next case

            # Add to subset_primary
            subset_primary[patient_id][case_id] = text

            # Add to subset_superset
            subset_superset[patient_id][case_id] = superset_cases[case_id]

    return subset_primary, subset_superset, missing_patients, missing_cases


def print_missing(missing_patients, missing_cases):
    """
    Print missing patients and cases.
    """
    if missing_patients:
        print("Missing Patients in Superset File:")
        for pid in missing_patients:
            print(f"  - {pid}")
    else:
        print("No missing patients in the superset file.")

    if missing_cases:
        print("\nMissing Cases in Superset File:")
        for pid, cases in missing_cases.items():
            print(f"  - Patient {pid}:")
            for cid in cases:
                print(f"      - {cid}")
    else:
        print("No missing cases in the superset file.")


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create subsets of patient reports from two JSON files."
    )
    parser.add_argument(
        "primary_file", help="Path to the primary JSON file (subset reference)."
    )
    parser.add_argument("superset_file", help="Path to the superset JSON file.")
    parser.add_argument(
        "--output_primary",
        default="subset_primary.json",
        help="Output path for the primary subset JSON file.",
    )
    parser.add_argument(
        "--output_superset",
        default="subset_superset.json",
        help="Output path for the superset subset JSON file.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load JSON files
    print(f"Loading primary file: {args.primary_file}")
    primary_data = load_json(args.primary_file)

    print(f"Loading superset file: {args.superset_file}")
    superset_data = load_json(args.superset_file)

    # Create subsets
    print("Creating subsets...")
    subset_primary, subset_superset, missing_patients, missing_cases = create_subset(
        primary_data, superset_data
    )

    # Save subsets
    print(f"Saving primary subset to: {args.output_primary}")
    save_json(subset_primary, args.output_primary)

    print(f"Saving superset subset to: {args.output_superset}")
    save_json(subset_superset, args.output_superset)

    # Print missing information
    print("\nChecking for missing patients and cases...")
    print_missing(missing_patients, missing_cases)

    print("\nSubset creation completed.")


if __name__ == "__main__":
    main()
