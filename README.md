# MOSAIC

**MOSAIC** (Multimodal Optical Slide Analysis Including Comparisons) is a framework for training and inferencing vision-language models for computational pathology. The code has been released as part of the paper "Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions" by Lucassen et al. (2025). The exact version of the code used in the paper is available in the `0.1.0` tag. Please note that this repository contains more content than described in the associated paper. Specifically, there are additional model definitions based on HIPT, including 'attention' model configurations that can be used with features extracted using HIPT.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-compatible GPU (recommended for training, but CPU is supported)

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/SanderMoon/MOSAIC.git
cd MOSAIC
```

2. Install the package:

```bash
pip install -e .
```

3. Install additional dependencies:

```bash
# Install pycocoevalcap manually (not available on PyPI)
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

### Development Installation

For development, install with additional development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Slide-Level Inference with Text Generation

For pathology slide analysis with text generation, you can process multiple slide feature files:

```python
from mosaic.model_factory import create_model, load_pretrained
import torch
import os

# Model configuration
model_name = "coca_stage_2_perceiver_frozen_uni"
pretrained_path = "checkpoints/mosaic-perceiver-biogpt-lora.pt"  # Update filename for other models 
device = "cpu"  # or "cuda" if available

# Create model and tokenizer
model, tokenizer, amp, input_dtype = create_model(
    model_name=model_name,
    pretrained=None,
    precision="bf16",
    device=device,
    init_tokenizer=True,
)

# Load pretrained weights
load_pretrained(model, pretrained=pretrained_path, device=device)

def load_features_from_pth(file_path: str) -> torch.Tensor:
    """
    Load features from a .pth file with nested dictionary structure.
    
    Returns:
        torch.Tensor: Features of shape [1, N, D] where N is number of patches
    """
    data = torch.load(file_path, map_location=device)
    features_list = []
    
    # Extract features from nested structure: {level: {patch_id: {'feature': tensor}}}
    for level_key in data.keys():
        level_data = data[level_key]
        for patch_id in sorted(level_data.keys()):
            if "feature" in level_data[patch_id]:
                feature = level_data[patch_id]["feature"]
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature)
                features_list.append(feature.to(device))
    
    if features_list:
        stacked_features = torch.stack(features_list, dim=0)
        return stacked_features.unsqueeze(0) 
    else:
        raise ValueError(f"No features found in {file_path}")

# Generation parameters
generation_params = {
    "seq_len": 128,
    "max_seq_len": 128,
    "temperature": 1.0,
    "generation_type": "top_k",
    "top_k": 1,
    "min_seq_len": 5,
    "repetition_penalty": 1.1,
}

# Process slide features
slide_path = "data/nevus_case_14710_0.pth"  # Example slide
visual_features = load_features_from_pth(slide_path)

model.eval()
with torch.no_grad():
    # Generate pathology report
    generated_ids = model.generate(
        image=visual_features,
        sot_token_id=tokenizer.all_special_ids[0],  # Start of text token
        eos_token_id=tokenizer.all_special_ids[1],  # End of text token
        pad_token_id=tokenizer.all_special_ids[3],  # Padding token
        **generation_params,
    )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated Report: {generated_text.strip()}")
```

## Pre-trained Models

Pre-trained MOSAIC models are available on Hugging Face at [SaltySander/MOSAIC](https://huggingface.co/SaltySander/MOSAIC). Three model variants are available:
- `mosaic-perceiver-biogpt-lora.pt` - LoRA fine-tuned model
- `mosaic-perceiver-biogpt-frozen.pt` - Frozen backbone model  
- `mosaic-perceiver-biogpt-unfrozen.pt` - Fully fine-tuned model

### Downloading Model Checkpoints

The models require access permission. Please request access to the repository first, then set your Hugging Face token:

```bash
# Set your Hugging Face token
export HF_TOKEN=your_huggingface_token_here

# Install huggingface_hub CLI
pip install huggingface_hub[cli]

# Download the LoRA model (change filename for other models)
huggingface-cli download SaltySander/MOSAIC checkpoints/mosaic-perceiver-biogpt-lora.pt --local-dir . --local-dir-use-symlinks False
```

Or download manually by visiting the [Hugging Face repository](https://huggingface.co/SaltySander/MOSAIC) and downloading the desired checkpoint file to your local `checkpoints/` directory.

## Training

### Basic Training Command

Here's an example training command with the main parameters:

```bash
python main.py \
    --model=coca_stage_2_perceiver_lora_uni \
    --pretrained=path/to/pretrained/model.pt \
    --train-split=path/to/train_ids.txt \
    --test-split=path/to/test_ids.txt \
    --val-split=path/to/val_ids.txt \
    --logs=path/to/logs \
    --text-data-file=path/to/reports.json \
    --root-dir=path/to/features \
    --log-local \
    --workers=8 \
    --batch-size=4 \
    --accum-freq=1 \
    --epochs=30 \
    --lr=1e-4 \
    --beta1=0.9 \
    --beta2=0.999 \
    --eps=1.0e-8 \
    --wd=1e-6 \
    --lr-scheduler=cosine \
    --warmup=600 \
    --precision=pure_bf16 \
    --image-features-cutoff=100000 \
    --report-to=tensorboard \
    --log-every-n-steps=1 \
    --seed=42 \
    --coca-caption-loss-weight=2.0 \
    --coca-contrastive-loss-weight=1.0 \
    --device=cuda \
    --dist-backend=nccl \
    --local-loss \
    --gather-with-grad \
    --save-frequency=1 \
    --val-frequency=1 \
    --caption-val-freq=1 \
    --eval-grace-period=5 \
    --caption-val-max-seq-len=256 \
    --val-gen-top-k=1 \
    --zsc-specimen-class-mapping path/to/class_mappings.json \
    --zsc-class-prompt-mapping path/to/prompt_mappings.json \
    --eval-metric-ci=0.95 \
    --eval-metric-bootstraps=1 \
    --test
```

For detailed training parameters and configuration options, see the [training documentation](docs/training.md).

## Documentation

This repository contains additional documentation to help you get started:

- **[Model Configurations](docs/model_configs.md)**: Learn how model configurations are structured and defined
- **[Dataset Structure](docs/dataset.md)**: Understand the required data structure for training
- **[Training Guide](docs/training.md)**: Detailed training parameters and options

## Citation

If you use this software in your research, please cite our paper:

### BibTeX

```bibtex
@misc{lucassen2025pathologyreportgenerationmultimodal,
    title={Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions},
    author={Ruben T. Lucassen and Sander P. J. Moonemans and Tijn van de Luijtgaarden and Gerben E. Breimer and Willeke A. M. Blokx and Mitko Veta},
    year={2025},
    eprint={2502.19293},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2502.19293},
}
```

### APA Style

Lucassen, R. T., Moonemans, S. P. J., van de Luijtgaarden, T., Breimer, G. E., Blokx, W. A. M., & Veta, M. (2025). *Pathology Report Generation and Multimodal Representation Learning for Cutaneous Melanocytic Lesions*. arXiv preprint arXiv:2502.19293.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## Contact

For questions or support, please contact:

- Sander Moonemans: <sander.moonemans@gmail.com>

---

*This work was developed as part of research into computational pathology and vision-language models for medical image analysis.*
