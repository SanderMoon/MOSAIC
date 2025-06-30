import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
import itertools
import random
import numpy as np


@dataclass
class ClassConfig:
    name: str
    num_samples: int
    image_background_color: str
    image_bar_orientation: str
    image_bar_thickness: str
    augment_images: bool
    augment_images_noise_level: float
    augment_images_zoom_factor: Tuple[float, float]
    augment_texts: bool


@dataclass
class DatasetConfig:
    hdf5_filename: str
    features_key: str = "features"
    positions_key: str = "positions"
    tile_keys_key: str = "tile_keys"
    text_key: str = "synthetic_text"


@dataclass
class SplitConfig:
    split: bool
    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Train, validation, and test ratios must sum to 1.0, but got {total}"
            )


@dataclass
class PreprocessingConfig:
    resize: bool = True
    resize_size: Optional[Tuple[int, int]] = None
    normalize: bool = True
    normalization_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    additional_transforms: Optional[dict] = None


@dataclass
class EmbedderConfig:
    embedder_name: str
    device: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass
class ExperimentConfig:
    class_configs: List[ClassConfig]
    embedder_config: EmbedderConfig
    dataset_config: DatasetConfig
    split_config: SplitConfig
    output_dir: str
    num_tiles_base: int = 1
    image_size: Tuple[int, int] = (224, 224)


class ConfigParser:
    """
    A parser class to read a JSON configuration file and convert it into an ExperimentConfig instance.
    Handles class templates and variations to generate multiple ClassConfig instances.
    """

    def __init__(self, json_path: str):
        """
        Initializes the ConfigParser with the path to the JSON configuration file.

        Parameters:
        - json_path: Path to the JSON configuration file.
        """
        self.json_path = json_path
        self.config_data = self._load_json()
        self.experiment_config = self._parse_config()

    def _load_json(self) -> Dict[str, Any]:
        """
        Loads the JSON configuration file.

        Returns:
        - Dictionary representation of the JSON configuration.

        Raises:
        - FileNotFoundError: If the JSON file does not exist.
        - json.JSONDecodeError: If the JSON file is malformed.
        """
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(
                f"Configuration file '{self.json_path}' does not exist."
            )

        with open(self.json_path, "r") as f:
            try:
                data = json.load(f)
                print(f"Successfully loaded configuration from '{self.json_path}'.")
                return data
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON file: {e}")

    def _parse_class_configs(
        self, class_template: Dict[str, Any], variations: Dict[str, List[Any]]
    ) -> List[ClassConfig]:
        """
        Generates ClassConfig instances based on a class template and specified variations.

        Parameters:
        - class_template: Dictionary representing the class template.
        - variations: Dictionary where keys are variation parameters and values are lists of their possible values.

        Returns:
        - List of ClassConfig instances.
        """
        class_configs = []

        # Generate all combinations of variations
        variation_keys = list(variations.keys())
        variation_values = list(variations.values())

        for combination in itertools.product(*variation_values):
            variation_dict = dict(zip(variation_keys, combination))

            # Merge class_template with variation_dict
            merged_config = {**class_template, **variation_dict}

            # Generate a unique class name based on variations
            variation_suffix = "_".join([str(v) for v in combination])
            class_name = f"P{random.randint(100000,999999)}_{variation_suffix}"

            class_config = ClassConfig(
                name=class_name,
                num_samples=merged_config.get("num_samples", 3),
                image_background_color=merged_config.get(
                    "image_background_color", "blue"
                ),
                image_bar_orientation=merged_config.get(
                    "image_bar_orientation", "horizontal"
                ),
                image_bar_thickness=merged_config.get("image_bar_thickness", "medium"),
                augment_images=merged_config.get("augment_images", True),
                augment_images_noise_level=merged_config.get(
                    "augment_images_noise_level", 10.0
                ),
                augment_images_zoom_factor=tuple(
                    merged_config.get("augment_images_zoom_factor", [1.0, 1.3])
                ),
                augment_texts=merged_config.get("augment_texts", True),
            )
            class_configs.append(class_config)
            print(f"Generated ClassConfig: {class_config}")

        return class_configs

    def _parse_embedder_config(
        self, embedder_config_data: Dict[str, Any]
    ) -> EmbedderConfig:
        """
        Parses the embedder configuration from JSON data.

        Parameters:
        - embedder_config_data: Dictionary representing embedder configuration.

        Returns:
        - EmbedderConfig instance.

        Raises:
        - KeyError: If required fields are missing in the embedder configuration.
        """
        try:
            preprocessing_data = embedder_config_data.get("preprocessing", {})
            preprocessing_config = PreprocessingConfig(
                resize=preprocessing_data.get("resize", True),
                resize_size=(
                    tuple(preprocessing_data["resize_size"])
                    if preprocessing_data.get("resize_size")
                    else None
                ),
                normalize=preprocessing_data.get("normalize", True),
                normalization_mean=tuple(
                    preprocessing_data.get("normalization_mean", (0.485, 0.456, 0.406))
                ),
                normalization_std=tuple(
                    preprocessing_data.get("normalization_std", (0.229, 0.224, 0.225))
                ),
                additional_transforms=preprocessing_data.get("additional_transforms"),
            )

            embedder_config = EmbedderConfig(
                embedder_name=embedder_config_data["embedder_name"],
                device=embedder_config_data["device"],
                preprocessing=preprocessing_config,
            )
            print(f"Parsed EmbedderConfig: {embedder_config}")
            return embedder_config
        except KeyError as e:
            raise KeyError(f"Missing key {e} in embedder_config")
        except Exception as e:
            raise ValueError(f"Error parsing embedder_config: {e}")

    def _parse_dataset_config(
        self, dataset_config_data: Dict[str, Any]
    ) -> DatasetConfig:
        """
        Parses the dataset configuration from JSON data.

        Parameters:
        - dataset_config_data: Dictionary representing dataset configuration.

        Returns:
        - DatasetConfig instance.

        Raises:
        - KeyError: If required fields are missing in the dataset configuration.
        """
        try:
            dataset_config = DatasetConfig(
                hdf5_filename=dataset_config_data["hdf5_filename"],
                features_key=dataset_config_data.get("features_key", "features"),
                positions_key=dataset_config_data.get("positions_key", "positions"),
                tile_keys_key=dataset_config_data.get("tile_keys_key", "tile_keys"),
                text_key=dataset_config_data.get("text_key", "synthetic_text"),
            )
            print(f"Parsed DatasetConfig: {dataset_config}")
            return dataset_config
        except KeyError as e:
            raise KeyError(f"Missing key {e} in dataset_config")
        except Exception as e:
            raise ValueError(f"Error parsing dataset_config: {e}")

    def _parse_split_config(self, split_config_data: Dict[str, Any]) -> SplitConfig:
        """
        Parses the split configuration from JSON data.

        Parameters:
        - split_config_data: Dictionary representing split configuration.

        Returns:
        - SplitConfig instance.

        Raises:
        - KeyError: If required fields are missing in the split configuration.
        - ValueError: If the split ratios do not sum to 1.0.
        """
        try:
            split_config = SplitConfig(
                split=split_config_data["split"],
                train=split_config_data["train"],
                val=split_config_data["val"],
                test=split_config_data["test"],
            )
            print(f"Parsed SplitConfig: {split_config}")
            return split_config
        except KeyError as e:
            raise KeyError(f"Missing key {e} in split_config")
        except Exception as e:
            raise ValueError(f"Error parsing split_config: {e}")

    def _parse_experiment_config(self) -> ExperimentConfig:
        """
        Parses the entire experiment configuration from JSON data.

        Returns:
        - ExperimentConfig instance.

        Raises:
        - KeyError: If required top-level fields are missing.
        """
        try:
            # Extract class template and variations
            class_template = self.config_data.get("class_template", {})
            variations = self.config_data.get("variations", {})

            if not class_template:
                raise ValueError("No 'class_template' found in configuration.")
            if not variations:
                raise ValueError("No 'variations' found in configuration.")

            # Parse variations to ensure they are lists
            variations = {k: v for k, v in variations.items() if isinstance(v, list)}
            if not variations:
                raise ValueError("No valid variation parameters found in 'variations'.")

            # Generate class_configs based on template and variations
            class_configs = self._parse_class_configs(class_template, variations)

            # Parse other configurations
            embedder_config_data = self.config_data["embedder_config"]
            embedder_config = self._parse_embedder_config(embedder_config_data)

            dataset_config_data = self.config_data["dataset_config"]
            dataset_config = self._parse_dataset_config(dataset_config_data)

            split_config_data = self.config_data["split_config"]
            split_config = self._parse_split_config(split_config_data)

            output_dir = self.config_data["output_dir"]
            num_tiles_base = self.config_data.get("num_tiles_base", 1)
            image_size = tuple(self.config_data.get("image_size", [224, 224]))

            experiment_config = ExperimentConfig(
                class_configs=class_configs,
                embedder_config=embedder_config,
                dataset_config=dataset_config,
                split_config=split_config,
                output_dir=output_dir,
                num_tiles_base=num_tiles_base,
                image_size=image_size,
            )
            print(
                f"Successfully parsed ExperimentConfig with {len(class_configs)} classes."
            )
            return experiment_config
        except KeyError as e:
            raise KeyError(f"Missing top-level key {e} in configuration")
        except Exception as e:
            raise ValueError(f"Error parsing ExperimentConfig: {e}")

    def _parse_config(self) -> ExperimentConfig:
        """
        Orchestrates the parsing of the JSON configuration into an ExperimentConfig instance.

        Returns:
        - ExperimentConfig instance.
        """
        return self._parse_experiment_config()

    def get_experiment_config(self) -> ExperimentConfig:
        """
        Retrieves the parsed ExperimentConfig instance.

        Returns:
        - ExperimentConfig instance.
        """
        return self.experiment_config
