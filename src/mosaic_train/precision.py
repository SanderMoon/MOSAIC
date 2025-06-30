from contextlib import suppress

import torch


def get_autocast(precision: str):
    if precision == "amp":
        return torch.autocast(device_type="cuda")
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        return suppress
