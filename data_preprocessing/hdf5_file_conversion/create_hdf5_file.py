#!/usr/bin/env python3

import argparse
from hdf5_file_manager import HDF5FileManager
import logging
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create and manage an organized HDF5 file from multiple .pth and JSON files."
    )
    parser.add_argument(
        "--hdf5_filename",
        type=str,
        default="hipt_text_dataset.h5",
        help="Name of the HDF5 file to create/manage.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="h5_logs",
        help="Directory where log files will be saved.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        nargs="+",  # Allows one or more arguments
        default=None,  # Changed to None to detect absence
        help="One or more root directories containing feature data.",
    )
    parser.add_argument(
        "--feature_attribute",
        type=str,
        default="hipt_features",
        help="Attribute name for feature data in HDF5.",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        nargs="+",  # Allows one or more arguments
        default=None,  # Changed to None to detect absence
        help="One or more paths to JSON files containing text data.",
    )
    parser.add_argument(
        "--text_attributes",
        type=str,
        nargs="+",  # Allows one or more arguments
        default=None,  # Changed to None to detect absence
        help="Attribute name for text data in HDF5.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Initialize the HDF5 file manager with logging
    manager = HDF5FileManager(hdf5_filename=args.hdf5_filename, log_dir=args.log_dir)

    # Add feature data for each root directory if provided
    if args.root_dir:
        for root_dir in args.root_dir:
            logging.info(
                f"Starting to add feature data from root directory: {root_dir}"
            )
            manager.add_feature_data(
                root_dir=root_dir, attribute_name=args.feature_attribute
            )
            logging.info(
                f"Completed adding feature data from root directory: {root_dir}"
            )
    else:
        logging.info("No root directories provided, skipping feature data addition.")

    # Add text data for each text file if provided
    if args.text_file and args.text_attributes:
        for text_file, text_attribute in zip(args.text_file, args.text_attributes):
            logging.info(f"Starting to add text data from file: {text_file}")
            manager.add_text_data(file_name=text_file, attribute_name=text_attribute)
            logging.info(f"Completed adding text data from file: {text_file}")
    else:
        logging.info(
            "No text files or text attributes provided, skipping text data addition."
        )

    # Retrieve and print total processed sizes
    total_size = manager.get_total_processed_size()
    total_pth_size = manager.get_total_pth_size()
    total_text_size = manager.get_total_text_size()

    print(f"Total size of all files processed: {total_size}")
    print(f"Total size of .pth files processed: {total_pth_size}")
    print(f"Total size of text files processed: {total_text_size}")

    manager.shutdown_logging()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
