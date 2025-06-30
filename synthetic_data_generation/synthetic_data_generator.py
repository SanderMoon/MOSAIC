import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List


@dataclass
class ClassConfig:
    name: str  # Class label (e.g., 'P000001_T20-58255II')
    num_samples: int  # Number of samples to generate for this class
    image_background_color: str  # Background color of the image (e.g., 'blue')
    image_bar_orientation: str  # Orientation of the bar in the image ('horizontal', 'vertical', 'diagonal')
    image_bar_thickness: (
        str  # Thickness of the bar in the image ('thin', 'medium', 'thick')
    )
    augment_images: bool  # Whether to augment images
    augment_images_noise_level: float  # Noise level for image augmentation (e.g., 10.0)
    augment_images_zoom_factor: Tuple[
        float, float
    ]  # Zoom factor range for image augmentation (min, max)
    augment_texts: bool  # Whether to augment text


class SyntheticDataGenerator:
    """
    A class to generate synthetic images with corresponding text descriptions.
    """

    def __init__(
        self, image_size=(256, 256), noise_level=10, base_image_size=(256, 256)
    ):
        """
        Initializes the data generator.

        Parameters:
        - image_size: Tuple specifying the size of the image (width, height).
        - noise_level: Standard deviation for Gaussian noise.
        - base_image_size: The base size for scaling thickness (default: (256, 256)).
        """
        self.image_size = image_size
        self.noise_level = noise_level
        self.base_image_size = base_image_size  # New attribute for scaling
        self.color_map = self._create_color_map()
        self.thickness_map = {"thin": 5, "medium": 20, "thick": 50}
        self.orientations = ["horizontal", "vertical", "diagonal"]

    def _create_color_map(self):
        """
        Creates a mapping from color names to RGB tuples.
        Extend this dictionary as needed.
        """
        return {
            "blue": (0, 0, 255),
            "red": (255, 0, 0),
            "green": (34, 139, 34),
            "yellow": (255, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "lime": (0, 255, 0),
            "pink": (255, 192, 203),
            "teal": (0, 128, 128),
            "lavender": (230, 230, 250),
            "brown": (165, 42, 42),
            "beige": (245, 245, 220),
            "maroon": (128, 0, 0),
            "navy": (0, 0, 128),
        }

    def _apply_noise(self, image_array, noise_level):
        """
        Applies Gaussian noise to the image.

        Parameters:
        - image_array: NumPy array of the image.
        - noise_level: Standard deviation for Gaussian noise.

        Returns:
        - Noisy image as a NumPy array.
        """
        noise = np.random.normal(0, noise_level, image_array.shape).astype(np.int16)
        noisy_image = image_array.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    def _draw_bar(self, draw, orientation, thickness, color=(0, 0, 0)):
        """
        Draws a black bar on the image with dynamically scaled thickness.

        Parameters:
        - draw: PIL ImageDraw object.
        - orientation: Orientation of the bar ('horizontal', 'vertical', 'diagonal').
        - thickness: Base thickness of the bar in pixels.
        - color: RGB tuple for the bar color.
        """
        width, height = self.image_size
        base_width, base_height = self.base_image_size
        scaling_factor = (
            width / base_width
        )  # Assuming square images; otherwise, consider separate scaling
        scaled_thickness = max(
            1, int(thickness * scaling_factor)
        )  # Ensure thickness is at least 1 pixel

        if orientation == "horizontal":
            y = height // 2
            draw.rectangle(
                [0, y - scaled_thickness // 2, width, y + scaled_thickness // 2],
                fill=color,
            )
        elif orientation == "vertical":
            x = width // 2
            draw.rectangle(
                [x - scaled_thickness // 2, 0, x + scaled_thickness // 2, height],
                fill=color,
            )
        elif orientation == "diagonal":
            draw.line([(0, 0), (width, height)], fill=color, width=scaled_thickness)
        else:
            raise ValueError(
                "Invalid orientation. Choose from 'horizontal', 'vertical', 'diagonal'."
            )

    def augment_image(self, image, zoom_factor_range: Tuple[float, float]):
        """
        Applies augmentations to the image, including zoom-in and brightness adjustment.

        Parameters:
        - image: PIL Image object.
        - zoom_factor_range: Tuple specifying (min_zoom, max_zoom) factors.

        Returns:
        - Augmented PIL Image object.
        """
        # Get the original size
        width, height = image.size

        # Apply zoom-in only (no zoom out)
        min_zoom, max_zoom = zoom_factor_range
        zoom_factor = random.uniform(min_zoom, max_zoom)
        new_size = (int(width * zoom_factor), int(height * zoom_factor))

        # Resize the image with zoom-in
        image = image.resize(new_size, Image.BICUBIC)

        # Crop to the original size to avoid black borders
        left = (new_size[0] - width) // 2
        top = (new_size[1] - height) // 2
        image = image.crop((left, top, left + width, top + height))

        # Apply random brightness changes
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(
            random.uniform(0.8, 1.2)
        )  # Adjust brightness between 0.8 and 1.2

        return image

    def generate_image(
        self, config: ClassConfig, image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Generates an image based on the specified class configuration, applies noise and augmentations.

        Parameters:
        - config: Instance of ClassConfig.
        - image_size: Optional tuple specifying the size of the image (width, height). If None, use self.image_size.

        Returns:
        - PIL Image object.
        """
        if image_size is None:
            image_size = self.image_size

        # Create background
        if config.image_background_color not in self.color_map:
            raise ValueError(
                f"Color '{config.image_background_color}' not recognized. Available colors: {list(self.color_map.keys())}"
            )
        bg_rgb = self.color_map[config.image_background_color]
        image = Image.new("RGB", image_size, bg_rgb)
        draw = ImageDraw.Draw(image)

        # Determine thickness
        if config.image_bar_thickness not in self.thickness_map:
            raise ValueError(
                f"Thickness '{config.image_bar_thickness}' not recognized. Choose from {list(self.thickness_map.keys())}"
            )
        thickness_px = self.thickness_map[config.image_bar_thickness]

        # Draw black bar with scaled thickness
        self._draw_bar(
            draw, config.image_bar_orientation, thickness_px, color=(0, 0, 0)
        )

        # Convert to NumPy array and apply noise
        image_array = np.array(image)
        noise_level = (
            config.augment_images_noise_level
            if config.augment_images
            else self.noise_level
        )
        noisy_image = self._apply_noise(image_array, noise_level)

        # Convert back to PIL Image
        final_image = Image.fromarray(noisy_image)

        # Apply augmentations
        if config.augment_images:
            final_image = self.augment_image(
                final_image, config.augment_images_zoom_factor
            )

        return final_image

    def generate_text(self, config: ClassConfig):
        """
        Generates a varied textual description based on the specified class configuration.

        Parameters:
        - config: Instance of ClassConfig.

        Returns:
        - String describing the image.
        """
        templates = [
            f"This image depicts a plain {config.image_background_color} background with a {config.image_bar_thickness} {config.image_bar_orientation} oriented black bar.",
            f"A {config.image_bar_thickness}, {config.image_bar_orientation} black bar is shown against a {config.image_background_color} backdrop.",
            f"On a {config.image_background_color} field, there is a {config.image_bar_thickness} black bar oriented {config.image_bar_orientation}.",
            f"The picture features a {config.image_background_color} background crossed by a {config.image_bar_thickness} {config.image_bar_orientation} black bar.",
        ]
        if config.augment_texts:
            return random.choice(templates)
        else:
            return templates[0]

    def generate_sample(
        self, config: ClassConfig, image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Generates a single image-text pair based on the class configuration.

        Parameters:
        - config: Instance of ClassConfig.
        - image_size: Optional tuple specifying the size of the image (width, height). If None, use self.image_size.

        Returns:
        - Tuple (PIL Image, String)
        """
        image = self.generate_image(config, image_size)
        text = self.generate_text(config)
        return image, text

    def generate_random_sample(
        self,
        class_configs: List[ClassConfig],
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Generates a random image-text pair by randomly selecting class configurations.

        Parameters:
        - class_configs: List of ClassConfig instances to choose from.
        - image_size: Optional tuple specifying the size of the image (width, height). If None, use self.image_size.

        Returns:
        - Tuple (PIL Image, String)
        """
        config = random.choice(class_configs)
        return self.generate_sample(config, image_size)

    def batch_generate(
        self,
        class_configs: List[ClassConfig],
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Generates a batch of image-text pairs based on class configurations.

        Parameters:
        - class_configs: List of ClassConfig instances.
        - image_size: Optional tuple specifying the size of the images (width, height). If None, use self.image_size.

        Returns:
        - List of tuples [(PIL Image, String), ...]
        """
        samples = []
        for config in class_configs:
            for _ in range(config.num_samples):
                image, text = self.generate_sample(config, image_size)
                samples.append((image, text))
        return samples
