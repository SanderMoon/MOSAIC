import os

import torch
from huggingface_hub import hf_hub_download


def get_model_path(model_name: str, cache_dir: str = ".model_cache"):
    """
    Retrieves the file path for a specified model, either from a local cache or by
    downloading it from the Hugging Face Hub.

    Args:
        model_name (str): The name or identifier of the model on the Hugging Face Hub.
        cache_dir (str, optional): The directory where cached model weights will be stored.
            Defaults to ".model_cache".

    Returns:
        str: The file path to the model weights (.pth file).
    """

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Construct the path for the cached .pth file
    pth_file = os.path.join(cache_dir, f"{model_name.replace('/', '_')}.pth")

    # Check if the .pth file is already cached
    if os.path.exists(pth_file):
        print(f"Loading cached weights from {pth_file}")
        return pth_file

    # Download the model checkpoint if not cached
    print(f"Downloading weights for {model_name}")
    checkpoint_file = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")

    # Load the state dictionary from the checkpoint
    state_dict = torch.load(checkpoint_file, map_location="cpu")

    # Save the state dictionary as a .pth file
    torch.save(state_dict, pth_file)
    print(f"Saved weights to {pth_file}")

    return pth_file
