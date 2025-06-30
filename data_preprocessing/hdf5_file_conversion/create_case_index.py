#!/usr/bin/env python3
"""
create_case_index.py

A one-off script to append a 'case-index' group to an existing HDF5 file.
This index maps case IDs to their HDF5 paths for efficient data retrieval.

Usage:
    python create_case_index.py --hdf5 your_data.hdf5
"""

import h5py
import logging
import argparse
import sys


def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Append a 'case-index' group to an HDF5 file."
    )
    parser.add_argument(
        "--hdf5", type=str, required=True, help="Path to the HDF5 file to be indexed."
    )
    return parser.parse_args()


def collect_case_ids_and_paths(hdf5_file):
    """
    Traverse the 'patients' group to collect case IDs and their HDF5 paths.

    Parameters:
        hdf5_file (h5py.File): Opened HDF5 file in read mode.

    Returns:
        tuple: Two lists containing case_ids and case_paths respectively.
    """
    case_ids = []
    case_paths = []

    if "patients" not in hdf5_file:
        logging.error("The HDF5 file does not contain a 'patients' group.")
        return case_ids, case_paths

    patients_group = hdf5_file["patients"]
    num_patients = len(patients_group)
    logging.info(f"Found 'patients' group with {num_patients} patients.")

    for idx, patient_id in enumerate(patients_group, 1):
        patient_group = patients_group[patient_id]
        num_cases = len(patient_group)
        logging.debug(f"Patient '{patient_id}' has {num_cases} cases.")

        for case_id in patient_group:
            case_path = f"/patients/{patient_id}/{case_id}"
            case_ids.append(case_id)
            case_paths.append(case_path)
            logging.debug(f"Collected case_id: {case_id}, path: {case_path}")

        if idx % 100 == 0 or idx == num_patients:
            logging.info(f"Processed {idx}/{num_patients} patients.")

    logging.info(f"Total cases collected: {len(case_ids)}")
    return case_ids, case_paths


def create_case_index_group(hdf5_filename, case_ids, case_paths):
    """
    Create or overwrite the 'case-index' group in the HDF5 file.

    Parameters:
        hdf5_filename (str): Path to the HDF5 file.
        case_ids (list): List of case IDs.
        case_paths (list): List of HDF5 paths corresponding to each case ID.
    """
    try:
        with h5py.File(hdf5_filename, "a") as hdf5_file:
            # If 'case-index' exists, delete it to overwrite
            if "case-index" in hdf5_file:
                del hdf5_file["case-index"]
                logging.info("Existing 'case-index' group found and deleted.")

            # Create 'case-index' group
            case_index_group = hdf5_file.create_group("case-index")
            logging.info("'case-index' group created.")

            # Define variable-length UTF-8 string data type
            string_dtype = h5py.string_dtype(encoding="utf-8")

            # Create 'case_ids' dataset
            case_index_group.create_dataset(
                "case_ids",
                data=case_ids,
                dtype=string_dtype,
                compression="gzip",
                compression_opts=9,
            )
            logging.info("'case_ids' dataset created with compression.")

            # Create 'case_paths' dataset
            case_index_group.create_dataset(
                "case_paths",
                data=case_paths,
                dtype=string_dtype,
                compression="gzip",
                compression_opts=9,
            )
            logging.info("'case_paths' dataset created with compression.")

        logging.info("Successfully created the 'case-index' group in the HDF5 file.")

    except Exception as e:
        logging.error(f"Failed to create 'case-index' group: {e}")
        sys.exit(1)


def main():
    """Main function to execute the script."""
    setup_logging()
    args = parse_arguments()

    hdf5_filename = args.hdf5
    logging.info(f"Starting to create 'case-index' for HDF5 file: {hdf5_filename}")

    try:
        with h5py.File(hdf5_filename, "r") as hdf5_file:
            case_ids, case_paths = collect_case_ids_and_paths(hdf5_file)

        if not case_ids:
            logging.error("No case IDs found. Exiting without creating 'case-index'.")
            sys.exit(1)

        create_case_index_group(hdf5_filename, case_ids, case_paths)

    except FileNotFoundError:
        logging.error(f"The file '{hdf5_filename}' does not exist.")
        sys.exit(1)
    except OSError as e:
        logging.error(f"Error opening the HDF5 file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logging.info("Script completed successfully.")


if __name__ == "__main__":
    main()
