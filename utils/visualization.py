import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data.preprocessing import dcm_to_numpy, rle2mask
from segmentation_models_pytorch.metrics import get_stats, iou_score, precision, recall, f1_score

def display_image_mask(image_path, row, isResizing=False):
    """ Displays an image and its combined masks.

    Images are stored as DICOM files and are converted to numpy arrays using dcm_to_numpy().
    The masks are encoded as RLE masks in .csv files and are decoded and combined to make one mask.

    Args:
        image_path: The path of a DICOM image.
        row: The rows of a .csv file that match the image ID.
        isResizing: Apply resizing or not to an image and mask pair.
    """

    # Parse .dcm file into numpy array
    image = dcm_to_numpy(image_path)
    
    # Initialize a blank numpy array that match the shape of the image
    masks = np.zeros(image.shape, dtype=np.uint8)
    for element in row:
        # Transform RLE into mask
        mask = rle2mask(element, image.shape[0], image.shape[1]).astype(np.uint8)
        
        # Combine mask into one single mask
        masks = np.add(masks, mask)
    
    # Convert pixels in mask as either 0 or 255
    ret, masks = cv2.threshold(masks, 0, 255, cv2.THRESH_BINARY)

    # Apply resizing to image and mask from 1024x1024 to 256x256
    if isResizing:  
        masks = cv2.resize(masks, (256, 256), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # Display image and mask and overlay
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(masks, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(masks, cmap="Reds", alpha=0.4)
    plt.axis("off")
    
    plt.show()

def plot_losses(train_losses, val_losses):
    """Display training and validation loss on a graph after training.

    Args:
    train_losses: A Python list that contains the training loss values for each epoch.
    val_losses: A Python list that contains the validation loss values for each epoch.
    """

    plt.figure(figsize=(5, 5))

    plt.title("Loss")
    plt.plot(train_losses,  label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()

    plt.show()

def show_batch_prediction(image, mask, pred_mask):
    """Retrieves a single batch from the test dataloader and displays a batch of the model's predicted masks along with the input image and truth mask.

    Args:
    image - The input image.
    mask - The truth mask.
    pred_mask - The model's predicted mask.
    """

    # [C, H, W] -> [H, W, C]
    # Move images from GPU to CPU and convert to numpy array for visualization
    image = image.permute(1, 2, 0).cpu().numpy()
    mask = mask.permute(1, 2, 0).cpu().numpy()
    pred_mask = pred_mask.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    plt.show()

def compute_metrics(model, dataloader, device):
    """ Displays the average IoU, precision, recall, and Dice score for the test dataset.

    Args:
    model - The trained model.
    dataloader - The test dataset.
    device - Converts image and mask to run on CUDA.
    """

    # Put model on evaluation mode
    model.eval()

    # Keep track of total scores
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_dice = 0

    # Disable gradients
    with torch.inference_mode():
        # Loop through each batch in the test dataset
        for _, (images, masks) in enumerate(tqdm(dataloader)):
            # Get a single batch of images and masks from test data
            images, masks = images.to(device), masks.to(device)

            # Get model's predicted masks
            pred_masks = model(images)

            # Squeeze model's predicted mask's pixels in [0, 1] range
            probs = torch.sigmoid(pred_masks)

            # Threshold pixels to 0 or 1
            pred_masks = (probs > 0.5).float()
            
            # Get confusion matrix stats
            tp, fp, fn, tn = get_stats(pred_masks, masks.to(torch.uint8), mode="binary", threshold=0.5)

            # Retrieve scores
            score_iou = iou_score(tp, fp, fn, tn, reduction="micro")
            score_precision = precision(tp, fp, fn, tn, reduction="micro")
            score_recall = recall(tp, fp, fn, tn, reduction="micro")
            score_dice = f1_score(tp, fp, fn, tn, reduction="micro")

            # Add scores after each batch
            total_iou += score_iou.item()
            total_precision += score_precision.item()
            total_recall += score_recall.item()
            total_dice += score_dice.item()
        
        # Display average scores
        print(f'IoU: {total_iou / len(dataloader)}')
        print(f'Precision: {total_precision / len(dataloader)}')
        print(f'Recall: {total_recall / len(dataloader)}')
        print(f'Dice: {total_dice / len(dataloader)}')