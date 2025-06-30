## Dataset Structure for `CaseDataset`

The Python code defines `CaseDataset` for loading image features and corresponding text data. For this class to function correctly, the dataset needs to be organized in a specific directory structure and be accompanied by key metadata and annotation files.

### I. Overall Directory Layout

The dataset is expected to be rooted in a main directory (referred to as `root_dir`). An optional supplementary directory, named by appending `_remainder` to the `root_dir` name (e.g., if `root_dir` is `hipt_superbatches`, then `hipt_superbatches_remainder`), can also be used to store additional data batches.

The basic layout is as follows:

```
<root_dir>/
├── data_0/
│   ├── extracted_features/
│   │   ├── example_feature_file_1.pth
│   │   └── ...
│   └── feature_information.txt
├── data_1/
│   ├── extracted_features/
│   │   ├── example_feature_file_2.pth
│   │   └── ...
│   └── feature_information.txt
└── ... (more data_N batch directories)

<root_dir>_remainder/  (Optional: if present, structured like <root_dir>)
├── data_X/
│   ├── extracted_features/
│   │   └── ...
│   └── feature_information.txt
└── ...
```

-   **`<root_dir>/`**: The primary path containing multiple batch directories.
-   **`<root_dir>_remainder/`**: An optional directory, conventionally named by appending `_remainder` to the main root directory's name. If it exists, `CaseDataset` will also scan it for batch directories.
-   **`data_N/`**: Individual batch directories (e.g., `data_0`, `data_1`). Each such directory contains:
    * `feature_information.txt`: A metadata file linking specimen information to feature files.
    * `extracted_features/`: A subdirectory containing the actual feature files (`.pth`).
    * *(Note: An initial comment in the provided code also mentions `tile_information.txt` within batch directories, but the `CaseDataset` class itself does not appear to utilize this specific file.)*

### II. Key Files and Their Formats

1.  **`feature_information.txt`** (located inside each `data_N/` directory)

    This text file acts as an index for the features within its batch directory. Each entry in this file consists of exactly three consecutive lines:

    * **Line 1**: A JSON string representing a list of original Whole Slide Image (WSI) names associated with the feature set.
        *Example:* `["WSI_filename_1.ndpi", "WSI_filename_2.svs"]`
    * **Line 2**: A JSON string containing metadata about the specimen.
        *Example:* `{"specimen_index": 123, "patient": "P001001", "specimen": "Tx-24-0001A", "size": 10.5}`
        Crucial fields here are:
        * `"patient"`: The patient identifier.
        * `"specimen"`: The specimen identifier, which `CaseDataset` uses as the `specimen_id` (effectively the `case_id` for matching).
        * `"specimen_index"`: An index for the specimen.
    * **Line 3**: The filename of the corresponding `.pth` feature file (located in the `extracted_features/` subdirectory), enclosed in single quotes.
        *Example:* `'feature_set_alpha.pth'`

    These three-line blocks repeat for every distinct feature set described in the file.

2.  **`.pth` Files** (located inside `extracted_features/` subdirectories)

    These are PyTorch files (`torch.load` compatible) containing the extracted image features and their positions. The structure of a `.pth` file is a dictionary where integer keys (e.g., `0`, `1`) represent different components or stages of feature extraction (e.g., from different HIPT model components). The `CaseDataset` is initialized with a `comp_index` to select which component's data to load.

    The structure for a given `comp_index` is another dictionary:

    ```python
    # Content of a .pth file, e.g., feature_set_alpha.pth
    {
        0: {  # Data for comp_index = 0
            # Key: tuple (specimen_index, slide_index, cross_section_index, tile_index)
            (123, 0, 0, 0): {
                "feature": torch.tensor([[...], [...]], dtype=torch.float32), # Shape: (num_sub_features, feature_dim1) e.g. (N, 384)
                "position": torch.tensor([x, y, z], dtype=torch.float32)    # Shape: (3)
            },
            (123, 0, 0, 1): {
                "feature": torch.tensor([[...]], dtype=torch.float32), # Shape: (num_sub_features, feature_dim1)
                "position": torch.tensor([x', y', z'], dtype=torch.float32)
            },
            # ... more entries for other tiles/regions
        },
        1: {  # Data for comp_index = 1
            (123, 0, 0, 0): {
                "feature": torch.tensor([[...]], dtype=torch.float32), # Shape: (num_sub_features, feature_dim2) e.g. (M, 192)
                "position": torch.tensor([x, y, z], dtype=torch.float32)
            },
            # ... more entries
        }
        # ... other component indices if present
    }
    ```

    * The outer dictionary keys (`0`, `1`, etc.) are selected by `CaseDataset`'s `comp_index` parameter.
    * The inner dictionary keys are tuples, typically `(specimen_index, slide_index, cross_section_index, tile_index)`, identifying unique tiles or regions.
    * `"feature"`: A PyTorch tensor holding the image features. `CaseDataset` stacks these, so the first dimension can vary.
    * `"position"`: A PyTorch tensor for the spatial coordinates or positional encoding related to the features.

3.  **`case_id_file`** (Path provided to `CaseDataset` constructor)

    This is a plain text file that lists the `case_ids` (which correspond to `specimen_id` from `feature_information.txt`) that should be included in the dataset. The `CaseDataset` reads this file, expects the IDs to be comma-separated if on a single line, and then processes them. IDs can also be on separate lines if the `split(',')` logic effectively isolates them after stripping whitespace.

    *Example (`case_ids_train.txt`):*
    ```
    Tx-24-0001A,Tx-24-0002B,Tx-24-0003C
    ```
    Or (also handled by current parsing logic after `strip()`):
    ```
    Tx-24-0001A,
    Tx-24-0002B,
    Tx-24-0003C
    ```

4.  **`text_data_file`** (Path provided to `CaseDataset` constructor)

    This is a JSON file containing text annotations (e.g., medical reports, notes). The `CaseDataset` expects this file to have a nested dictionary structure: `{"patient_id": {"specimen_id": "text_annotation_string"}}`.

    *Example (`text_annotations.json`):*
    ```json
    {
        "P001001": {
            "Tx-24-0001A": "This is the clinical report for specimen Tx-24-0001A of patient P001001...",
            "Tx-24-0005E": "Follow-up notes for specimen Tx-24-0005E..."
        },
        "P001002": {
            "Tx-24-0002B": "Pathology findings for Tx-24-0002B..."
        }
    }
    ```

