import torch


class PadImageBatchTo8NPlus1:
    """Pad an IMAGE batch to the smallest length matching multiplier*n+remainder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiplier": (
                    "INT",
                    {"default": 8, "min": 1, "max": 999, "step": 1},
                ),
                "remainder": (
                    "INT",
                    {"default": 1, "min": 0, "max": 999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "original_batch_length")
    FUNCTION = "pad_batch"
    CATEGORY = '😺dzNodes/LayerUtility'

    def pad_batch(
        self,
        images: torch.Tensor,
        multiplier: int = 8,
        remainder: int = 1,
    ):
        original_length = len(images)
        if original_length == 0:
            raise ValueError("Cannot pad an empty IMAGE batch.")
        if multiplier < 1:
            raise ValueError("Multiplier must be at least 1.")
        if remainder < 0 or remainder >= multiplier:
            raise ValueError(
                "Remainder must satisfy 0 <= remainder < multiplier "
                f"({remainder} is invalid for multiplier {multiplier})."
            )

        n = max(
            0,
            (original_length - remainder + multiplier - 1) // multiplier,
        )
        target_length = multiplier * n + remainder
        padding_length = target_length - original_length

        if padding_length == 0:
            return (images, original_length)

        repeated_last_frame = images[-1:].expand(
            padding_length, *images.shape[1:]
        )
        padded_images = torch.cat((images, repeated_last_frame), dim=0)
        return (padded_images, original_length)


class RestoreImageBatchLength:
    """Crop a processed IMAGE batch back to its original length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_batch_length": (
                    "INT",
                    {"forceInput": True},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "restore_batch"
    CATEGORY = '😺dzNodes/LayerUtility'

    def restore_batch(self, images: torch.Tensor, original_batch_length: int):
        current_length = len(images)
        if original_batch_length < 1:
            raise ValueError("Original batch length must be at least 1.")
        if original_batch_length > current_length:
            raise ValueError(
                "Original batch length cannot exceed the input IMAGE batch length "
                f"({original_batch_length} > {current_length})."
            )

        return (images[:original_batch_length],)


NODE_CLASS_MAPPINGS = {
    "LayerUtility: Pad Image Batch to 8n+1": PadImageBatchTo8NPlus1,
    "LayerUtility: Restore Pad Image Batch": RestoreImageBatchLength,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayerUtility: PadImageBatch": "LayerUtility: Pad Image Batch to 8n+1",
    "LayerUtility: RestorePadImageBatch": "LayerUtility: Restore Pad Image Batch",
}
