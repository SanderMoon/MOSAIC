import logging
from dataclasses import MISSING, Field, fields
from typing import Any, ClassVar, Dict, Protocol, Type, TypeVar, cast

import torch


class DataclassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field]]


T = TypeVar("T", bound=DataclassProtocol)

logger = logging.getLogger(__name__)


def filter_dataclass_fields(
    dataclass_type: Type[T], data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filters a dictionary to include only the fields defined in the given dataclass.

    Args:
        dataclass_type (Type[T]): The dataclass type to use for filtering.
        data (Dict[str, Any]): The dictionary to filter.

    Returns:
        Dict[str, Any]: A new dictionary containing only the valid fields for the dataclass.
    """
    valid_fields = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data.items() if k in valid_fields}


def create_dataclass_instance(dataclass_type: Type[T], data: Dict[str, Any]) -> T:
    """
    Creates an instance of a dataclass, filling in missing fields with default values if available.
    Issues a warning for missing fields and extra fields.

    Args:
        dataclass_type (Type[T]): The dataclass type to instantiate.
        data (Dict[str, Any]): The dictionary of data to use for instantiation.

    Returns:
        T: An instance of the dataclass.
    """
    dataclass_fields = fields(dataclass_type)
    field_names = set(f.name for f in dataclass_fields)

    # Warn about extra fields
    extra_fields = set(data.keys()) - field_names
    if extra_fields:
        logger.warning(
            f"Extra fields in data for {dataclass_type.__name__}: {extra_fields}"
        )

    init_kwargs = {}
    for field in dataclass_fields:
        field_name = field.name
        if field_name in data:
            # if data field is of type dict
            if isinstance(data[field_name], dict) and field_name != "extra_config":
                init_kwargs[field_name] = create_dataclass_instance(
                    cast(Type[T], field.type), data[field_name]
                )
            else:
                init_kwargs[field_name] = data[field_name]
        else:
            if field.default != MISSING:
                # Default value is available
                init_kwargs[field_name] = field.default
                logger.warning(
                    f"Missing field '{field_name}' in data for {dataclass_type.__name__}, using default value {field.default}"
                )
            elif field.default_factory != MISSING:
                # Default factory is available
                init_kwargs[field_name] = field.default_factory()
                logger.warning(
                    f"Missing field '{field_name}' in data for {dataclass_type.__name__}, using default_factory value"
                )
            else:
                # No default value, field is required
                logger.warning(
                    f"Missing required field '{field_name}' in data for {dataclass_type.__name__}, setting it to None"
                )
                init_kwargs[field_name] = None

    return dataclass_type(**init_kwargs)


def filter_config_fields(config_type: Type[T], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters a dictionary to include only the fields valid for the given config type.

    Args:
        config_type (Type[T]): The config type to use for filtering.
        data (Dict[str, Any]): The dictionary to filter

    Returns:
        Dict[str, Any]: A new dictionary containing only the valid fields for the config
    """
    # Get the default config
    default_config = config_type()

    # Use the __dict__ of the default config to determine valid fields
    valid_fields = set(default_config.__dict__.keys())

    return {k: v for k, v in data.items() if k in valid_fields}


def create_config_instance(config_type: Type[T], data: Dict[str, Any]) -> T:
    """
    Creates an instance of a config, ignoring any extra fields in the input data.

    Args:
        config_type (Type[T]): The config type to instantiate
        data (Dict[str, Any]): The dictionary of data to use for instantiation

    Returns:
        T: An instance of the config
    """
    filtered_data = filter_config_fields(config_type, data)
    return config_type(**filtered_data)


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,),
        ),
    )
    model.visual.image_size = image_size
    return model


def _token_to_tensor(token_id, device: str = "cpu") -> torch.Tensor:
    if not isinstance(token_id, torch.Tensor):
        if isinstance(token_id, int):
            token_id = [token_id]
        token_id = torch.tensor(token_id, device=device)
    return token_id
