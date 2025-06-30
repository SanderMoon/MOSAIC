import torch
from transformers import AutoModel, AutoFeatureExtractor
from config import EmbedderConfig
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEmbedder:
    """
    A class to embed images using a HuggingFace Vision Transformer model.
    It preprocesses images, generates embeddings, and saves visualizations.
    """

    def __init__(self, config: EmbedderConfig):
        """
        Initializes the ImageEmbedder with a specified configuration.

        Parameters:
        - config: Instance of EmbedderConfig containing embedder and preprocessing settings.
        """
        self.config = config
        self.model_name = self.config.embedder_name
        self.device = (
            self.config.device
            if self.config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            f"Initializing ImageEmbedder with model '{self.model_name}' on device '{self.device}'"
        )

        try:
            # Load the feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name
            )
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise

        # Determine expected image size from the feature extractor
        self.expected_size = self._determine_expected_size()
        logger.info(f"Expected image size for embedding: {self.expected_size}")

    def _determine_expected_size(self):
        """
        Determines the expected image size from the feature extractor.

        Returns:
        - Tuple representing (width, height)
        """
        if hasattr(self.feature_extractor, "size"):
            size_attr = self.feature_extractor.size
            if isinstance(size_attr, dict):
                height = size_attr.get("height", 224)
                width = size_attr.get("width", 224)
                return (width, height)
            elif isinstance(size_attr, int):
                return (size_attr, size_attr)
            elif isinstance(size_attr, tuple) and len(size_attr) == 2:
                return size_attr
            else:
                logger.warning(
                    f"Unexpected 'size' attribute format: {size_attr}. Using default size (224, 224)."
                )
        else:
            logger.warning(
                "No 'size' attribute found in feature extractor. Using default size (224, 224)."
            )
        return (224, 224)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocesses a PIL Image for embedding based on the configuration.

        Parameters:
        - image: PIL Image object.

        Returns:
        - Preprocessed tensor ready for the model.
        """
        try:
            # Apply resizing if enabled
            if self.config.preprocessing.resize:
                target_size = (
                    self.config.preprocessing.resize_size or self.expected_size
                )
                image = image.resize(target_size, Image.BICUBIC)
                logger.debug(f"Resized image to {target_size}.")
            else:
                logger.debug("Resizing is disabled.")

            # Remove normalization
            inputs = self.feature_extractor(
                images=image, return_tensors="pt", normalize=False
            )
            logger.debug("Normalization is removed.")

            return inputs
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            raise

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Generates an embedding for a given image.

        Parameters:
        - image: PIL Image object.

        Returns:
        - Numpy array representing the image embedding.
        """
        try:
            inputs = self.preprocess_image(image)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token representation as the embedding
                if (
                    hasattr(outputs, "pooler_output")
                    and outputs.pooler_output is not None
                ):
                    embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
                else:
                    # Fallback to mean pooling if pooler_output is not available
                    embedding = (
                        outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                    )

            logger.info("Image embedded successfully.")
            return embedding
        except Exception as e:
            logger.error(f"Error during image embedding: {e}")
            raise

    def save_preprocessed_visualization(
        self, original_image: Image.Image, save_path: str
    ):
        """
        Saves a side-by-side visualization of the original and preprocessed images.

        Parameters:
        - original_image: PIL Image object.
        - save_path: Path to save the visualization image.
        """
        try:
            # Apply preprocessing to get the preprocessed image
            preprocessed_inputs = self.preprocess_image(original_image)

            # Extract the preprocessed image for visualization
            preprocessed_tensor = preprocessed_inputs["pixel_values"].squeeze(0).cpu()
            preprocessed_image = preprocessed_tensor.clone()

            # Convert to numpy and transpose to (H, W, C)
            preprocessed_image = preprocessed_image.numpy().transpose(1, 2, 0)

            # If normalization was applied, you'd need to reverse it here
            # Since normalization is removed, we can proceed directly
            preprocessed_image = np.clip(preprocessed_image, 0, 1)

            # Convert to an 8-bit image (values 0-255)
            preprocessed_image = (preprocessed_image * 255).astype(np.uint8)

            # Convert to PIL image
            preprocessed_image_pil = Image.fromarray(preprocessed_image)

            # Resize original image to match preprocessed image size for visualization
            original_resized = original_image.resize(
                preprocessed_image_pil.size, Image.BICUBIC
            )

            # Create a side-by-side image
            combined_width = original_resized.width + preprocessed_image_pil.width
            combined_height = max(
                original_resized.height, preprocessed_image_pil.height
            )
            combined_image = Image.new("RGB", (combined_width, combined_height))
            combined_image.paste(original_resized, (0, 0))
            combined_image.paste(preprocessed_image_pil, (original_resized.width, 0))

            # Add labels to each image using matplotlib
            plt.figure(figsize=(10, 5))
            plt.imshow(combined_image)
            plt.axis("off")
            plt.title("Original Image (Left) vs. Preprocessed Image (Right)")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved preprocessed visualization to {save_path}")
        except Exception as e:
            logger.error(f"Error during visualization saving: {e}")
            raise
