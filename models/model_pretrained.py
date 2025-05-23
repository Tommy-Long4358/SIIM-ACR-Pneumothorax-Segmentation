import segmentation_models_pytorch as smp

def initialize_model(encoder_name, in_channels, out_channels, device):
    """Initializes a pretrained UNet model.

    The encoder can be switched to any pretrained model for experimenting.
    
    Args:
    encoder_name - The pretrained model for the encoder.
    in_channels - The number of channels (RGB or grayscale) as input.
    out_channels - The total number of classes to output (binary or multiclass)
    device - The device to run the model on (CPU or CUDA).

    Returns:
    The initialized model.
    """
    # Initialize UNet with pretrained encoder (efficientnet-b2) and imagenet weights
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels
    ).to(device)

    return model