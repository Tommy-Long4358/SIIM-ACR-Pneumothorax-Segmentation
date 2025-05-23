import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data.preprocessing import dcm_to_numpy, rle2mask
from segmentation_models_pytorch.metrics import get_stats, iou_score, precision, recall, f1_score

def display_image_mask(image_path, row, isResizing=False):
    # Parse .dcm file into numpy array
    image = dcm_to_numpy(image_path)
    
    masks = np.zeros(image.shape, dtype=np.uint8)
    for element in row:
        # Transform RLE into mask
        mask = rle2mask(element, image.shape[0], image.shape[1]).astype(np.uint8)
        masks = np.add(masks, mask)
    
    ret, masks = cv2.threshold(masks, 0, 255, cv2.THRESH_BINARY)

    if isResizing:  
        masks = cv2.resize(masks, (256, 256), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    print(image_path.stem)

    print(f'Image size: {image.shape}')
    print(f'Pixel range: {image.min(), image.max()}')
    print(image.dtype)
    print(masks.dtype)

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
    plt.figure(figsize=(5, 5))

    plt.title("Loss")
    plt.plot(train_losses,  label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()

    plt.show()

def show_batch_prediction(image, mask, pred_mask):    
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
    model.eval()

    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_dice = 0

    # Disable gradients
    with torch.inference_mode():
        for _, (images, masks) in enumerate(tqdm(dataloader)):
            # Get a single batch of images and masks from test data
            images, masks = images.to(device), masks.to(device)

            pred_masks = model(images)

            probs = torch.sigmoid(pred_masks)
            pred_masks = (probs > 0.5).float()
            
            tp, fp, fn, tn = get_stats(pred_masks, masks.to(torch.uint8), mode="binary", threshold=0.5)

            score_iou = iou_score(tp, fp, fn, tn, reduction="micro")
            score_precision = precision(tp, fp, fn, tn, reduction="micro")
            score_recall = recall(tp, fp, fn, tn, reduction="micro")
            score_dice = f1_score(tp, fp, fn, tn, reduction="micro")

            total_iou += score_iou.item()
            total_precision += score_precision.item()
            total_recall += score_recall.item()
            total_dice += score_dice.item()
        
        print(f'IoU: {total_iou / len(dataloader)}')
        print(f'Precision: {total_precision / len(dataloader)}')
        print(f'Recall: {total_recall / len(dataloader)}')
        print(f'Dice: {total_dice / len(dataloader)}')