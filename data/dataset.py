import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ChestDataset(Dataset):
    def __init__(self, images, masks, transform_images=None, transform_masks=None):
        self.images = images
        self.masks = masks
        self.transform_images = transform_images
        self.transform_masks = transform_masks

    def load_image(self, index):
        return Image.open(self.images[index]).convert("RGB")
    
    def load_mask(self, index):
        return Image.open(self.masks[index]).convert("L")
    
    def get_image_title(self, index):
        return self.images[index].stem

    def get_mask_title(self, index):
        return self.masks[index].stem 
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load image and mask and convert to numpy array
        image = np.array(self.load_image(index), dtype=np.float32) / 255.0
        mask = np.array(self.load_mask(index), dtype=np.float32) / 255.0

        # Apply data augmentation
        if self.transform_images:
            augment = self.transform_images(image=image, mask=mask)

            image = augment["image"]
            mask = augment["mask"]
        
        return image, mask.unsqueeze(0)