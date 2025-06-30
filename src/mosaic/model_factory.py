import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoTokenizer

from mosaic.custom_coca import CustomCLIP, CustomCoCa
from mosaic.custom_lm import CustomLM
from mosaic.model_configs import MultimodalConfig, TextConfig, VisionConfig
from mosaic.utils import create_dataclass_instance

logger = logging.getLogger(__name__)

HF_HUB_PREFIX = "hf-hub:"


def load_pretrained(
    model: nn.Module, pretrained: str, device: Union[str, torch.device]
):
    """Loads pretrained weights, handling various checkpoint formats.

    Args:
        model: The model to load weights into.
        pretrained: Path to the pretrained checkpoint.
        device: The device to load the checkpoint onto.
    """
    try:
        checkpoint = torch.load(pretrained, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {pretrained}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, OrderedDict) or isinstance(
        checkpoint, dict
    ):  # If it is just a state dict
        state_dict = checkpoint
    else:
        raise RuntimeError(
            "Checkpoint format not recognized. Expected 'state_dict' or a direct state_dict."
        )

    load_state_dict_robust(model, state_dict)
    print(f"Loaded pretrained weights from {pretrained}")


def load_state_dict_robust(model: nn.Module, state_dict: Dict, strict: bool = True):
    """Loads state_dict in a robust way, handling DataParallel/DistributedDataParallel wrappers.

    Args:
        model: The model to load the state_dict into.
        state_dict: The state_dict to load.
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model.state_dict().
    """
    new_state_dict = OrderedDict()
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    is_wrapped = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    )

    for k, v in state_dict.items():
        name = k
        if has_module_prefix and not is_wrapped:
            name = k[7:] if k.startswith("module.") else k  # Remove 'module.' prefix
        elif not has_module_prefix and is_wrapped:
            name = "module." + k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=strict)


def create_model(
    model_name: str,
    pretrained: Optional[str] = None,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    init_tokenizer: Optional[bool] = False,
) -> Tuple[nn.Module, Any, bool, torch.dtype]:
    """
    Creates and configures the CoCa model based on the specified precision.

    Args:
        model_name (str): Name of the model to create.
        pretrained (Optional[str], optional): Path or identifier for pretrained weights. Defaults to None.
        precision (str, optional): Desired precision setting. Defaults to 'fp32'.
        device (Union[str, torch.device], optional): Device to map the model to. Defaults to 'cpu'.
        jit (bool, optional): Whether to use TorchScript JIT. Defaults to False.
        init_tokenizer (Optional[str], optional): Whether to initialize the tokenizer. Defaults to False.
        **model_kwargs: Additional keyword arguments.

    Returns:
        Tuple[CustomCoCa, Any, bool, torch.dtype]: The configured model, tokenizer, AMP flag, and input dtype.
    """

    if model_name.startswith("coca"):
        model_type = "coca"
    elif model_name.startswith("clip"):
        model_type = "clip"
    elif model_name.startswith("lm"):
        model_type = "lm"
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if isinstance(device, str):
        device = torch.device(device)

    vision_config, text_config, multimodal_config = _get_configs(model_name, model_type)
    tokenizer = None
    if init_tokenizer:
        if not text_config.hf_tokenizer_name:
            raise ValueError(
                "hf_tokenizer_name not found in text_config. Please provide hf_tokenizer_name"
            )
        tokenizer = AutoTokenizer.from_pretrained(text_config.hf_tokenizer_name)
        if text_config.cls_token_id:
            _maybe_add_cls_token(tokenizer, True)
            text_config.cls_token_id = tokenizer.cls_token_id

    if model_type == "clip":
        model = CustomCLIP(vision_config, text_config)
    elif model_type == "coca":
        model = CustomCoCa(vision_config, text_config, multimodal_config)
    elif model_type == "lm":
        model = CustomLM(text_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Configure precision
    model, amp, input_dtype = configure_precision(model, precision, device)

    # If pretrained weights are specified, load them
    if pretrained:
        load_pretrained(model, pretrained, device)

    return model, tokenizer, amp, input_dtype


def _maybe_add_cls_token(tokenizer: Any, add_cls_token: bool) -> Any:
    """
    Adds a CLS token to the tokenizer if it does not already exist.
    Args:
        tokenizer: The tokenizer to modify.
        add_cls_token (bool): Whether to add the CLS token.

    Returns:
    Any: The modified tokenizer.
    """

    if add_cls_token:
        special_tokens_dict = {"cls_token": "[CLS]"}
        tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def _get_configs(
    model_name: str, model_type: str
) -> Tuple[VisionConfig, TextConfig, MultimodalConfig]:
    """
    Loads the configuration files for the specified model.
    Args:
        model_name (str): The name of the model.
        model_type (str): The type of the model (e.g., "coca", "clip", "lm").
    Returns:
    Tuple[VisionConfig, TextConfig, MultimodalConfig]: The vision, text, and multimodal configurations.

    """
    # Get the directory of the current file (model_factory.py)
    current_file = Path(__file__).resolve()

    # Construct the path to the model_configs directory
    model_config_dir = current_file.parent / "model_configs"

    # Construct the full path to the config file
    config_file = model_config_dir / f"{model_name}.json"

    # Check if the file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Read the config file
    with config_file.open() as f:
        all_config = json.load(f)

    vision_config = all_config.get("vision_cfg", {})
    text_config = all_config.get("text_cfg", {})

    if model_type == "coca" and "multimodal_cfg" not in all_config:
        raise ValueError(
            "multimodal_cfg not found in config file, please provide a multimodal_cfg (see config documentation for details)"
        )

    multimodal_config = all_config.get("multimodal_cfg", {})

    # Add any other params of all_config to the vision, text and multimodal config as well
    for key, value in all_config.items():
        if key not in ["vision_cfg", "text_cfg", "multimodal_cfg"]:
            vision_config[key] = value
            text_config[key] = value
            multimodal_config[key] = value

    return (
        _consolidate_vision_config(vision_config),
        _consolidate_text_config(text_config),
        _consolidate_multimodal_config(multimodal_config),
    )


def _consolidate_vision_config(config: Dict[str, Any]) -> VisionConfig:
    if config != {} and config.get("method") not in [
        "vit",
        "perceiver",
        "longnet",
        "pooler_only",
    ]:
        config["method"] = "vit"
        logger.warning(
            "Invalid or no method found in vision_config. Using default method 'vit'."
        )

    if config.get("method") == "pooler_only" and "pooler_only_config" not in config:
        logger.warning(
            "No pooler config found in vision_config. Using default config. (see config documentation for details)"
        )

    return create_dataclass_instance(VisionConfig, config)


def _consolidate_text_config(config: Dict[str, Any]) -> TextConfig:
    if config.get("load_pretrained") and "hf_model_name" not in config:
        raise ValueError(
            "hf_model_name not found in text_config. Please provide hf_model_name or set load_pretrained to False"
        )
    if "hf_tokenizer_name" not in config:
        raise ValueError(
            "hf_tokenizer_name not found in text_config. Please provide hf_tokenizer_name"
        )

    return create_dataclass_instance(TextConfig, config)


def _consolidate_multimodal_config(config: Dict[str, Any]) -> MultimodalConfig:
    if config.get("load_pretrained") and "hf_model_name" not in config:
        raise ValueError(
            "hf_model_name not found in multimodal_config. Please provide hf_model_name or set load_pretrained to False"
        )

    return create_dataclass_instance(MultimodalConfig, config)


def count_parameters(model: nn.Module) -> int:
    """
    Counts the number of trainable parameters in a PyTorch model

    Args:
        model (nn.Module): The PyTorch model

    Returns:
        int: The total number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_precision(
    model: nn.Module, precision: str, device: Union[str, torch.device] = "cpu"
) -> Tuple[nn.Module, bool, torch.dtype]:
    """
    Configures the model's dtype and determines if AMP should be used based on the precision argument.

    Args:
        model (nn.Module): The PyTorch model to configure.
        precision (str): The desired precision setting.
        device (Union[str, torch.device], optional): The device to map the model to. Defaults to 'cpu'.

    Returns:
        Tuple[nn.Module, bool, torch.dtype]:
            - Configured model.
            - Flag indicating whether to use AMP.
            - The dtype to which inputs should be cast.
    """
    # Define mapping from precision string to dtype and AMP usage
    precision = precision.lower()
    amp = False
    input_dtype = torch.float32  # Default input dtype

    if precision in ["fp32"]:
        # Standard precision
        model = model.to(device, dtype=torch.float32)
        input_dtype = torch.float32

    elif precision in ["fp16", "pure_fp16"]:
        if "pure" in precision:
            # Pure FP16: Cast model to FP16
            model = model.to(device, dtype=torch.float16)
        else:
            # For AMP variants, keep model in FP32
            model = model.to(device, dtype=torch.float32)
            amp = True
        input_dtype = torch.float16

    elif precision in ["bf16", "pure_bf16"]:
        if "pure" in precision:
            # Pure BF16: Cast model to BF16
            model = model.to(device, dtype=torch.bfloat16)
        else:
            # For AMP variants, keep model in FP32
            model = model.to(device, dtype=torch.float32)
            amp = True
        input_dtype = torch.bfloat16

    elif precision in ["amp", "amp_fp16", "amp_bf16", "amp_bfloat16"]:
        # AMP with optional specific lower precision
        amp = True
        model = model.to(device, dtype=torch.float32)  # Keep model in FP32 for AMP
        if "fp16" in precision:
            input_dtype = torch.float16
        elif "bf16" in precision or "bfloat16" in precision:
            input_dtype = torch.bfloat16
        else:
            input_dtype = torch.float32

    else:
        raise ValueError(f"Unsupported precision type: {precision}")

    return model, amp, input_dtype
