from mosaic.model_factory import create_model, load_pretrained
import torch
import os

# Specify the configuration name
model_name = "coca_stage_2_perceiver_lora_uni"
pretrained = None
precision = "bf16"
device = "cpu"

# Create the model and tokenizer
model, tokenizer, amp, input_dtype = create_model(
    model_name=model_name,
    pretrained=pretrained,
    precision=precision,
    device=device,
    init_tokenizer=True,
)

load_pretrained(
    model, pretrained="checkpoints/mosaic-perceiver-biogpt-lora.pt", device=device
)

text_inputs = tokenizer(
    "",
    return_tensors="pt",
    add_special_tokens=False,
    padding=True,
    truncation=True,
)

# empty attention mask
text_inputs["attention_mask"] = torch.ones_like(text_inputs["input_ids"])

data_dir = "data"
file_names = [
    "nevus_case_14710_0.pth",
]


def load_features_from_pth(file_path):
    """
    Load features from a .pth file with nested dictionary structure.

    Args:
        file_path (str): Path to the .pth file

    Returns:
        torch.Tensor: Tensor of shape [1, N, D] where N is number of embeddings
                     and D is the feature dimension
    """
    # Load the dictionary from the file
    data = torch.load(file_path, map_location=device)

    # Extract features from the nested dictionary structure
    # The structure is: {level: {patch_id: {'feature': tensor}}}
    features_list = []

    # Iterate through the outer dictionary (levels)
    for level_key in data.keys():
        level_data = data[level_key]

        # Iterate through patches in this level
        for patch_id in sorted(level_data.keys()):  # Sort to ensure consistent ordering
            patch_data = level_data[patch_id]

            # Extract the feature tensor
            if "feature" in patch_data:
                feature = patch_data["feature"]
                # Ensure it's a tensor and move to correct device
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature)
                feature = feature.to(device)
                features_list.append(feature)

    # Stack all features into a single tensor
    if features_list:
        # Stack features: [N, D]
        stacked_features = torch.stack(features_list, dim=0)
        # Add batch dimension: [1, N, D]
        batched_features = stacked_features.unsqueeze(0)
        return batched_features
    else:
        raise ValueError(f"No features found in {file_path}")


# Load visual inputs for all files
visual_input = []
for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)
    print(f"Loading features from {file_path}...")

    try:
        features = load_features_from_pth(file_path)
        visual_input.append(features)
        print(f"  Loaded tensor with shape: {features.shape}")
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        continue

print(f"\nLoaded {len(visual_input)} visual inputs successfully.")

# Set model to evaluation mode
model.eval()

# Generation parameters
generation_params = {
    "seq_len": 128,  # Maximum sequence length to generate
    "max_seq_len": 128,  # Maximum context length
    "temperature": 1,  # Controls randomness (lower = more deterministic)
    "generation_type": "top_k",  # Options: "beam_search", "top_p", "top_k"
    "top_k": 1,  # For top_k sampling
    "top_p": 0.9,  # For nucleus sampling
    "num_beams": 4,  # Number of beams for beam search
    "num_beam_groups": 2,  # Number of beam groups for diverse beam search
    "min_seq_len": 5,  # Minimum sequence length
    "repetition_penalty": 1.1,  # Penalty for repetition
}

# Process each visual input using generate function
with torch.no_grad():
    for i, visual_features in enumerate(visual_input):
        print(f"\n--- Generating text for file {i + 1}: {file_names[i]} ---")
        print(f"Visual features shape: {visual_features.shape}")

        try:
            # Generate text using the model's generate function
            generated_ids = model.generate(
                image=visual_features,
                sot_token_id=tokenizer.all_special_ids[0],
                eos_token_id=tokenizer.all_special_ids[1],
                pad_token_id=tokenizer.all_special_ids[3],
                **generation_params,
            )

            print(f"Generated token IDs shape: {generated_ids.shape}")

            # Decode the generated tokens to text
            if generated_ids.dim() > 1:
                # If batch dimension exists, process each sequence
                for batch_idx in range(generated_ids.shape[0]):
                    sequence = generated_ids[batch_idx]
                    decoded_text = tokenizer.decode(sequence, skip_special_tokens=True)
                    print(
                        f"Batch {batch_idx} - Generated text: '{decoded_text.strip()}'"
                    )
            else:
                # Single sequence
                decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print(f"Generated text: '{decoded_text.strip()}'")

        except Exception as e:
            print(f"Error during generation for {file_names[i]}: {e}")
            import traceback

            traceback.print_exc()
            continue

print("\nInference completed for all files.")
