import torch
import numpy as np
import cv2

def create_image_from_color(width, height, color="#FFFFFF"):
    """Create a solid color image with specified dimensions."""
    if isinstance(color, str) and color.startswith('#'):
        # Convert hex color to BGR (OpenCV format)
        color = tuple(int(color[i:i+2], 16) for i in (5, 3, 1))[::-1]
    return np.full((height, width, 3), color, dtype=np.uint8)

class AddMaskForICLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_image": ("IMAGE",),
                "patch_mode": (["grid", "patch_right", "patch_bottom"], {
                    "default": "grid",
                }),
                "patch_number": ([2, 3, 4], {
                    "default": 4,
                }),
                "patch_color": (["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"], {
                    "default": "#FF0000",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "x_offset", "y_offset", "target_width", "target_height", "total_width", "total_height")
    FUNCTION = "add_mask"
    OUTPUT_NODE = True
    CATEGORY = "ICLoraUtils/AddMaskForICLora"

    def add_mask(self, first_image, patch_mode, patch_number, patch_color):
        # Get the first image from batch
        image = first_image[0]
        height, width = image.shape[:2]

        # Calculate output dimensions
        target_width = width
        output_width = width * patch_number
        # Ensure output width is divisible by 64
        output_width = output_width - (output_width % 64) if output_width % 64 != 0 else output_width

        # Create colored patch
        colored_patch = create_image_from_color(target_width, height, patch_color)
        colored_patch = torch.from_numpy(colored_patch)

        # Create base masks
        image_mask = torch.zeros((height, width))
        patch_mask = torch.ones((height, width))

        # Concatenate images and masks horizontally
        patches = [image] + [colored_patch] * (patch_number - 1)
        masks = [image_mask] + [patch_mask] * (patch_number - 1)
        
        concatenated_image = np.hstack(patches)
        concatenated_mask = np.hstack(masks)

        # Calculate x offset (min_x)
        x_offset = int((100 / patch_number) / 100.0 * concatenated_image.shape[1])
        y_offset = 0

        # Normalize image to [0, 1] range
        concatenated_image = np.clip(255. * concatenated_image, 0, 255).astype(np.float32) / 255.0
        
        # Convert to torch tensors with batch dimension
        return_images = torch.from_numpy(concatenated_image)[None,]
        return_masks = torch.from_numpy(concatenated_mask)[None,]

        return (
            return_images, 
            return_masks, 
            x_offset, 
            y_offset, 
            target_width, 
            height, 
            concatenated_image.shape[1], 
            concatenated_image.shape[0]
        )

NODE_CLASS_MAPPINGS = {
    "AddMaskForICLora": AddMaskForICLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddMaskForICLora": "Add Mask For IC Lora x",
}
