import os
import torch
import h5py
import json
import numpy as np
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import ast
import gc

import concurrent.futures
from multiprocessing import cpu_count, Queue

from tqdm import tqdm
import threading


def worker_init(log_queue):
    """
    Initialize logging for worker processes.
    """
    queue_handler = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove any existing handlers
    logger.handlers = []
    logger.addHandler(queue_handler)


def process_entry_with_retry(entry, max_retries=3):
    """
    Wraps the process_entry function with retry logic.
    """
    for attempt in range(max_retries):
        try:
            data = process_entry(entry)
            return data
        except Exception as e:
            logging.error(f"Error processing entry {entry}: {e}")
            if attempt < max_retries - 1:
                logging.info(
                    f"Retrying entry {entry}. Attempt {attempt + 1}/{max_retries}"
                )
            else:
                logging.error(
                    f"Failed to process entry {entry} after {max_retries} attempts"
                )
                raise


def process_entry(entry):
    """
    Worker function to process a single entry and return the processed data.

    Parameters:
        entry (dict): A dictionary containing case information and the path to the feature file.

    Returns:
        data_to_send (dict): The processed data to be written to the HDF5 file.
    """
    try:
        # Set up logging in the worker process if necessary
        # This should be done at the module level or in an initializer
        # For example, you can configure logging to output to a file specific to each process

        logging.info(f"Processing feature file: {entry['feature_file_path']}")

        feature_file_path = entry["feature_file_path"]
        file_size = os.path.getsize(feature_file_path)
        logging.info(
            f"Processing feature file: {feature_file_path} (Size: {file_size} bytes)"
        )

        feature_data = torch.load(feature_file_path)
        # Prepare data to return
        case_info = entry["case_info"]
        patient_id = case_info["patient"]
        case_id = case_info["specimen"]

        data_to_send = {"patient_id": patient_id, "case_id": case_id, "components": []}

        for component_key, component_data in feature_data.items():
            features_list, positions_list, tile_keys_list = [], [], []
            for tile_key, tile_data in component_data.items():
                feature = np.array(tile_data["feature"])
                position = np.array(tile_data["position"])
                features_list.append(feature)
                positions_list.append(position)
                tile_keys_list.append(str(tile_key).encode("utf-8"))

            features_array = np.array(features_list)
            positions_array = np.array(positions_list)
            tile_keys_array = np.array(tile_keys_list, dtype="S")

            data_to_send["components"].append(
                {
                    "component_key": component_key,
                    "features": features_array,
                    "positions": positions_array,
                    "tile_keys": tile_keys_array,
                }
            )

            # Free memory used in this component
            del (
                features_list,
                positions_list,
                tile_keys_list,
                features_array,
                positions_array,
                tile_keys_array,
            )
            gc.collect()

        # Free memory used in feature_data
        del feature_data
        gc.collect()

        logging.info(f"Completed processing case {case_id}")

        return data_to_send

    except Exception as e:
        logging.error(f"Error processing entry {entry}: {e}")
        raise  # Re-raise the exception to be caught by the retry mechanism


class HDF5FileManager:
    def __init__(self, hdf5_filename, log_dir=None):
        """
        Initialize the HDF5FileManager.
        """
        self.hdf5_filename = hdf5_filename

        # Initialize total size trackers
        self.total_pth_size = 0  # Total size of .pth files processed
        self.total_text_size = 0  # Total size of text files processed (optional)
        self.total_size = 0  # Total size of all files processed

        # Set up logging configuration
        self.log_queue = Queue()
        self._setup_logging(log_dir)

        self.log_dir = log_dir

        # Ensure the HDF5 file exists
        if not os.path.exists(self.hdf5_filename):
            with h5py.File(self.hdf5_filename, "w") as hdf5_file:
                hdf5_file.create_group("patients")
                hdf5_file.create_group("case-index")
            logging.info(f"Created new HDF5 file: {self.hdf5_filename}")

    def _setup_logging(self, log_dir):
        """
        Set up logging configuration with a QueueListener.
        """
        # Create log directory if it doesn't exist
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "hdf5_file_manager.log")
        else:
            log_file = "hdf5_file_manager.log"

        # Define handlers
        file_handler = RotatingFileHandler(log_file, maxBytes=10**6, backupCount=5)
        console_handler = logging.StreamHandler()

        # Define a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Set up a listener for the queue
        listener = QueueListener(self.log_queue, file_handler, console_handler)
        listener.start()

        # Configure the root logger to use the queue
        queue_handler = QueueHandler(self.log_queue)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(queue_handler)

        # Store the listener and queue_handler so they can be stopped and removed later
        self.listener = listener
        self.queue_handler = queue_handler

        logging.info(f"Logging initialized. Log file: {log_file}")

    def shutdown_logging(self):
        """
        Stop the logging listener and clean up logging handlers.
        """
        self.listener.stop()  # Stop the QueueListener
        logger = logging.getLogger()
        logger.removeHandler(self.queue_handler)  # Remove the QueueHandler
        logging.shutdown()  # Shutdown the logging system

    @staticmethod
    def _human_readable_size(size, decimal_places=2):
        """
        Convert a size in bytes to a human-readable format.

        Parameters:
            size (int): Size in bytes.
            decimal_places (int): Number of decimal places for formatting.

        Returns:
            str: Human-readable size string.
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0 or unit == "TB":
                break
            size /= 1024.0
        return f"{size:.{decimal_places}f} {unit}"

    def load_text_data(self, file_name, attribute_name):
        """
        Load text data from a JSON file.

        Parameters:
            file_name (str): The path to the JSON file.
            attribute_name (str): The attribute name under which to store the text data.

        Returns:
            dict: A nested dictionary containing text data organized by attribute names.
        """
        text_data = {}
        if not os.path.exists(file_name):
            logging.warning(f"Text data file {file_name} does not exist.")
            return text_data

        file_size = os.path.getsize(file_name)
        self.total_text_size += file_size
        self.total_size += file_size
        logging.info(
            f"Processing text file: {file_name} ({self._human_readable_size(file_size)})"
        )

        with open(file_name, "r") as f:
            data = json.load(f)
        text_data[attribute_name] = data
        logging.info(
            f"Loaded text data from {file_name} under attribute '{attribute_name}'"
        )
        logging.info(
            f"Total text files size processed: {self._human_readable_size(self.total_text_size)}"
        )
        logging.info(
            f"Total files size processed: {self._human_readable_size(self.total_size)}"
        )
        return text_data

    def process_feature_information(self, feature_info_file):
        """
        Process the feature_information.txt file and extract entries.

        Parameters:
            feature_info_file (str): Path to the feature_information.txt file.

        Returns:
            list: A list of dictionaries containing slide names, case info, and feature filenames.
        """
        if not os.path.exists(feature_info_file):
            logging.warning(
                f"Feature information file {feature_info_file} does not exist."
            )
            return []

        file_size = os.path.getsize(feature_info_file)
        self.total_text_size += file_size
        self.total_size += file_size
        logging.info(
            f"Processing feature information file: {feature_info_file} ({self._human_readable_size(file_size)})"
        )

        with open(feature_info_file, "r") as f:
            lines = f.readlines()
        entries = []
        num_entries = len(lines) // 3
        for i in range(num_entries):
            idx = i * 3
            slide_names_line = lines[idx].strip()
            case_info_line = lines[idx + 1].strip()
            feature_filename_line = lines[idx + 2].strip()

            # Parse slide names using ast.literal_eval to handle single quotes
            try:
                slide_names = ast.literal_eval(slide_names_line)
                if not isinstance(slide_names, list):
                    raise ValueError("Slide names are not a list.")
            except (ValueError, SyntaxError) as e:
                logging.error(
                    f"Error parsing slide names on lines {idx+1}-{idx+3}: {e}"
                )
                continue

            # Parse case info using json.loads (assumes valid JSON)
            try:
                case_info = json.loads(case_info_line)
                if not isinstance(case_info, dict):
                    raise ValueError("Case info is not a dictionary.")
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Error parsing case info on lines {idx+2}-{idx+4}: {e}")
                continue

            # Parse feature filename (strip double quotes if present)
            feature_filename = feature_filename_line.strip().strip('"').strip("'")

            entry = {
                "slide_names": slide_names,
                "case_info": case_info,
                "feature_filename": feature_filename,
            }
            entries.append(entry)
        logging.info(f"Processed {len(entries)} entries from {feature_info_file}")
        logging.info(
            f"Total text files size processed: {self._human_readable_size(self.total_text_size)}"
        )
        logging.info(
            f"Total files size processed: {self._human_readable_size(self.total_size)}"
        )
        return entries

    def write_data_to_hdf5(self, data, attribute_name):
        """
        Writes the processed data to the HDF5 file.

        Parameters:
            data (dict): The data returned by process_entry.
            attribute_name (str): The attribute name under which to store the data.
        """
        try:
            with h5py.File(self.hdf5_filename, "a") as hdf5_file:
                patients_group = hdf5_file.require_group("patients")
                patient_id = data["patient_id"]
                case_id = data["case_id"]

                # Create or get the patient group
                patient_group = patients_group.require_group(patient_id)

                # Create or get the case group under the patient
                case_group = patient_group.require_group(case_id)

                # Store features and positions for each component
                for component in data["components"]:
                    component_key = component["component_key"]
                    features_array = component["features"]
                    positions_array = component["positions"]
                    tile_keys_array = component["tile_keys"]

                    component_group_name = f"{attribute_name}_comp_{component_key}"
                    component_group = case_group.require_group(component_group_name)

                    # Clear existing datasets if they exist
                    for dataset_name in ["features", "positions", "tile_keys"]:
                        if dataset_name in component_group:
                            del component_group[dataset_name]

                    # Store datasets with chunking and compression
                    component_group.create_dataset(
                        "features", data=features_array, chunks=True, compression="gzip"
                    )
                    component_group.create_dataset(
                        "positions",
                        data=positions_array,
                        chunks=True,
                        compression="gzip",
                    )
                    component_group.create_dataset(
                        "tile_keys",
                        data=tile_keys_array,
                        chunks=True,
                        compression="gzip",
                    )

                    # Free memory used in this component
                    del features_array, positions_array, tile_keys_array
                    gc.collect()

                logging.info(f"Successfully wrote data for case {case_id} to HDF5.")
        except Exception as e:
            logging.error(f"Error writing data to HDF5 for case {data['case_id']}: {e}")
            # Handle the exception as needed (e.g., re-raise, log, or pass)

    def add_feature_data(
        self, root_dir="data/hipt_superbatches", attribute_name="hipt_features"
    ):
        """
        Add feature data to the HDF5 file using ProcessPoolExecutor and handle retries.
        """
        logging.info(f"Starting to add feature data from root directory: {root_dir}")

        # Collect all entries to be processed
        entries = self.collect_entries(root_dir)

        if not entries:
            logging.warning("No entries found to process.")
            return

        total_entries = len(entries)
        processed_entries = 0
        failed_entries = []

        # Start the process pool executor
        num_workers = cpu_count() if cpu_count() > 2 else 1
        logging.info(
            f"Starting ProcessPoolExecutor with {num_workers} worker processes."
        )

        # Define the maximum number of concurrent futures
        max_concurrent_futures = (
            num_workers * 2
        )  # Adjust based on memory and performance

        semaphore = threading.Semaphore(max_concurrent_futures)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, initializer=worker_init, initargs=(self.log_queue,)
        ) as executor:
            future_to_entry = {}

            def callback(future):
                nonlocal processed_entries
                entry = future_to_entry.pop(future)
                semaphore.release()
                try:
                    data = future.result()
                    if data is not None:
                        try:
                            self.write_data_to_hdf5(data, attribute_name)
                            processed_entries += 1
                        except Exception as e:
                            logging.error(
                                f"Error writing data to HDF5 for case {data.get('case_id', 'unknown')}: {e}"
                            )
                            failed_entries.append(entry)
                    else:
                        failed_entries.append(entry)
                except Exception as exc:
                    logging.error(f"Entry {entry} generated an exception: {exc}")
                    failed_entries.append(entry)
                pbar.update(1)

            with tqdm(total=total_entries, desc="Processing entries") as pbar:
                for entry in entries:
                    semaphore.acquire()
                    future = executor.submit(process_entry_with_retry, entry)
                    future_to_entry[future] = entry
                    future.add_done_callback(callback)

            # Wait for all futures to complete
            executor.shutdown(wait=True)

        # Log the completion message **before** shutting down logging
        logging.info(
            f"Processing complete. Processed: {processed_entries}, Failed: {len(failed_entries)}"
        )

    def store_case_index(self, case_ids_list, case_paths_list):
        try:
            with h5py.File(self.hdf5_filename, "a") as hdf5_file:
                case_index_group = hdf5_file.require_group("case-index")

                # Update 'case_ids' dataset
                existing_case_ids = set(
                    [
                        cid.decode("utf-8") if isinstance(cid, bytes) else cid
                        for cid in case_index_group.get("case_ids", [])
                    ]
                )
                new_case_ids = [
                    cid for cid in case_ids_list if cid not in existing_case_ids
                ]
                if new_case_ids:
                    if "case_ids" in case_index_group:
                        case_ids_ds = case_index_group["case_ids"]
                        current_size = case_ids_ds.shape[0]
                        new_size = current_size + len(new_case_ids)
                        case_ids_ds.resize((new_size,))
                        case_ids_ds[current_size:new_size] = np.array(
                            new_case_ids, dtype="S"
                        )
                    else:
                        case_index_group.create_dataset(
                            "case_ids",
                            data=np.array(new_case_ids, dtype="S"),
                            maxshape=(None,),
                            chunks=True,
                        )
                    logging.info(
                        f"Updated 'case_ids' dataset with {len(new_case_ids)} entries."
                    )
                else:
                    logging.info("No new case_ids to append.")

                # Update 'case_paths' dataset similarly
                existing_case_paths = set(
                    [
                        cpath.decode("utf-8") if isinstance(cpath, bytes) else cpath
                        for cpath in case_index_group.get("case_paths", [])
                    ]
                )
                new_case_paths = [
                    cpath
                    for cpath in case_paths_list
                    if cpath not in existing_case_paths
                ]
                if new_case_paths:
                    if "case_paths" in case_index_group:
                        case_paths_ds = case_index_group["case_paths"]
                        current_size = case_paths_ds.shape[0]
                        new_size = current_size + len(new_case_paths)
                        case_paths_ds.resize((new_size,))
                        case_paths_ds[current_size:new_size] = np.array(
                            new_case_paths, dtype="S"
                        )
                    else:
                        case_index_group.create_dataset(
                            "case_paths",
                            data=np.array(new_case_paths, dtype="S"),
                            maxshape=(None,),
                            chunks=True,
                        )
                    logging.info(
                        f"Updated 'case_paths' dataset with {len(new_case_paths)} entries."
                    )
                else:
                    logging.info("No new case_paths to append.")
        except Exception as e:
            logging.error(f"Failed to update 'case-index' in HDF5 file: {e}")

    def log_failures(self, failed_entries):
        if failed_entries:
            logging.warning(f"Total failed entries: {len(failed_entries)}")
            for entry in failed_entries:
                try:
                    file_size = os.path.getsize(entry["feature_file_path"])
                    readable_size = self._human_readable_size(file_size)
                except Exception:
                    readable_size = "Unknown"
                logging.warning(f"Skipped Entry: {entry}, File Size: {readable_size}")

    def collect_entries(self, root_dir):
        entries = []
        for data_dir in os.listdir(root_dir):
            data_path = os.path.join(root_dir, data_dir)
            if os.path.isdir(data_path):
                feature_info_file = os.path.join(data_path, "feature_information.txt")
                extracted_features_dir = os.path.join(data_path, "extracted_features")
                if os.path.exists(extracted_features_dir):
                    if os.path.exists(feature_info_file):
                        entries_in_dir = self.process_feature_information(
                            feature_info_file
                        )
                        for entry in entries_in_dir:
                            feature_filename = entry["feature_filename"]
                            feature_file_path = os.path.join(
                                extracted_features_dir, feature_filename
                            )
                            if os.path.exists(feature_file_path):
                                entry["feature_file_path"] = feature_file_path
                                entries.append(entry)
                            else:
                                logging.warning(
                                    f"Feature file {feature_file_path} does not exist."
                                )
        return entries

    def add_text_data(self, file_name, attribute_name):
        """
        Add text data to the HDF5 file.

        Parameters:
            file_name (str): The path to the JSON file containing text data.
            attribute_name (str): The attribute name under which to store the text data.
        """
        # Load text data
        text_data = self.load_text_data(file_name, attribute_name)

        if not text_data:
            logging.warning("No text data loaded. Skipping text data addition.")
            return

        # Open the HDF5 file in append mode
        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            patients_group = hdf5_file["patients"]

            cases_processed = 0

            # Iterate over patients and cases
            for patient_id in patients_group:
                patient_group = patients_group[patient_id]
                for case_id in patient_group:
                    case_info = {"patient": patient_id, "specimen": case_id}
                    entry = {"case_info": case_info}

                    # Add text data to HDF5
                    self._add_text_data_to_hdf5(entry, hdf5_file, text_data)
                    cases_processed += 1
                    logging.info(
                        f"Added text data to patient {patient_id}, case {case_id} "
                        f"(Total cases processed: {cases_processed})"
                    )
                    logging.info(
                        f"Total files size processed: {self._human_readable_size(self.total_size)}"
                    )

        logging.info("Text data added to HDF5 file successfully.")
        logging.info(
            f"Total files size processed: {self._human_readable_size(self.total_size)}"
        )

    def _add_text_data_to_hdf5(self, entry, hdf5_file, text_data):
        """
        Internal method to add text data to the HDF5 file.

        Parameters:
            entry (dict): A dictionary containing case information.
            hdf5_file (h5py.File): The HDF5 file object.
            text_data (dict): The text data loaded from JSON files.
        """
        case_info = entry["case_info"]
        patient_id = case_info["patient"]
        case_id = case_info["specimen"]

        # Navigate to the case group
        patients_group = hdf5_file["patients"]
        if patient_id in patients_group:
            patient_group = patients_group[patient_id]
            if case_id in patient_group:
                case_group = patient_group[case_id]
                # Add text data attributes/datasets
                for attr_name, data in text_data.items():
                    if patient_id in data and case_id in data[patient_id]:
                        report_text = data[patient_id][case_id]
                        # Clean the attribute name to remove special characters
                        attr_name_clean = (
                            attr_name.replace("&", "_")
                            .replace("+", "plus")
                            .replace(" ", "_")
                        )
                        # Store as dataset
                        dataset_name = attr_name_clean.encode("utf-8")
                        # Remove existing dataset if it exists
                        if dataset_name in case_group:
                            del case_group[dataset_name]
                        # Assuming report_text is a string, store it as bytes
                        case_group.create_dataset(
                            dataset_name, data=report_text.encode("utf-8")
                        )
                        logging.info(
                            f"Added text data '{attr_name_clean}' to patient {patient_id}, case {case_id}"
                        )
                    else:
                        logging.warning(
                            f"No text data for patient {patient_id}, case {case_id} under attribute {attr_name}."
                        )

    def remove_feature_data(self):
        """
        Remove feature data from the HDF5 file.
        """
        # Open the HDF5 file in append mode
        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            # Remove patients group if it exists
            if "patients" in hdf5_file:
                del hdf5_file["patients"]
                hdf5_file.create_group("patients")
            # Clear 'case-index' group
            if "case-index" in hdf5_file:
                del hdf5_file["case-index"]
                hdf5_file.create_group("case-index")

        # Reset size trackers
        self.total_pth_size = 0
        self.total_text_size = 0
        self.total_size = 0

        logging.info("Feature data removed from HDF5 file successfully.")
        logging.info("Reset total size counters.")

    def remove_text_data(self, attributes=None):
        """
        Remove text data from the HDF5 file.

        Parameters:
            attributes (list or None): A list of attribute names to remove. If None, all text data will be removed.
        """
        # Open the HDF5 file in append mode
        with h5py.File(self.hdf5_filename, "a") as hdf5_file:
            patients_group = hdf5_file["patients"]

            # Iterate over patients and cases
            for patient_id in patients_group:
                patient_group = patients_group[patient_id]
                for case_id in patient_group:
                    case_group = patient_group[case_id]
                    datasets_to_remove = []
                    if attributes is None:
                        # Remove all datasets that are not 'component_*'
                        for dataset_name in case_group:
                            if not dataset_name.decode("utf-8").startswith(
                                "component_"
                            ):
                                datasets_to_remove.append(dataset_name)
                    else:
                        # Remove specified attributes
                        for attr_name in attributes:
                            attr_name_clean = (
                                attr_name.replace("&", "_")
                                .replace("+", "plus")
                                .replace(" ", "_")
                                .encode("utf-8")
                            )
                            if attr_name_clean in case_group:
                                datasets_to_remove.append(attr_name_clean)

                    for dataset_name in datasets_to_remove:
                        del case_group[dataset_name]
                        logging.info(
                            f"Removed dataset '{dataset_name.decode('utf-8')}' from patient {patient_id}, case {case_id}"
                        )

        logging.info("Text data removed from HDF5 file successfully.")
        logging.info(
            f"Total files size processed: {self._human_readable_size(self.total_size)}"
        )

    def get_case_index(self):
        """
        Retrieve the case index from the HDF5 file.

        Returns:
            dict: A dictionary mapping case IDs to case paths.
        """
        with h5py.File(self.hdf5_filename, "r") as hdf5_file:
            case_index_group = hdf5_file["case-index"]
            case_ids = case_index_group["case_ids"][:]
            case_paths = case_index_group["case_paths"][:]
            case_index = {}
            for case_id, case_path in zip(case_ids, case_paths):
                case_index[case_id.decode("utf-8")] = case_path.decode("utf-8")
            logging.info("Retrieved case index from HDF5 file")
            return case_index

    def get_total_processed_size(self):
        """
        Get the total size of all files processed.

        Returns:
            str: Total size in a human-readable format.
        """
        logging.info(
            f"Total files size processed: {self._human_readable_size(self.total_size)}"
        )
        return self._human_readable_size(self.total_size)

    def get_total_pth_size(self):
        """
        Get the total size of all .pth files processed.

        Returns:
            str: Total size in a human-readable format.
        """
        logging.info(
            f"Total .pth files size processed: {self._human_readable_size(self.total_pth_size)}"
        )
        return self._human_readable_size(self.total_pth_size)

    def get_total_text_size(self):
        """
        Get the total size of all text files processed.

        Returns:
            str: Total size in a human-readable format.
        """
        logging.info(
            f"Total text files size processed: {self._human_readable_size(self.total_text_size)}"
        )
        return self._human_readable_size(self.total_text_size)
