import numpy as np
import os
import cv2
from pydicom import dcmread
from tqdm import tqdm
from PIL import Image

def rle2mask(rle, width, height):
    """Converts RLE (Run-Length Encoded) masks into numpy arrays.
    Masks are initialized as 1D numpy arrays that are of length width * height.
    RLEs are a string that consists of numbers. Every odd number in the string represents which pixel to start at and every
    even number represents how many pixels from the odd number to turn on.
    
    Args:
    rle - A string of numbers.
    width - The dimension of the image.
    height - The dimension of the image.

    Returns:
    A 2D numpy array of the RLE mask.
    """
    # Create a 1D array of zeros that is of length width * height
    mask = np.zeros(width * height, dtype=np.float32)

    if rle == " -1":
        return mask.reshape((height, width), order='F')
    
    # Separate each number in the string by spaces and convert them to type int
    array = np.asarray([int(x) for x in rle.split()])

    # Get all index positions
    starts = array[0::2]

    # Get all pixels
    lengths = array[1::2]

    # Loop through each index position
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start

        # Convert elements from current position to current position + length index from 0 to 255
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    # RLE flattens 2D images in column-major order. It is similar to np.transpose
    return mask.reshape((height, width)).T

def dcm_to_numpy(image):
    """Converts a DICOM file into a numpy array with dcmread().

    Args:
    image - The file path for the DICOM file

    Returns:
    A 2D numpy array of the image converted.
    """
    image = dcmread(image)
    return image.pixel_array

def preprocess_data(output_dir, images, rle_pd):
    """Converts each image and mask into numpy arrays and resizes them to be saved as PNGs.

    It converts images represented as DICOM files into numpy arrays and translates RLE masks.
    
    Args:
    output_dir - The directory to save the images in.
    images - A 1D list of image paths
    rle_pd - A pandas .csv file of RLE masks for each image. An image can have multiple RLE masks.
    """
    os.makedirs(f"{output_dir}/pneumothorax/image", exist_ok=True)
    os.makedirs(f"{output_dir}/pneumothorax/mask", exist_ok=True)
    os.makedirs(f"{output_dir}/normal/image", exist_ok=True)
    os.makedirs(f"{output_dir}/normal/mask", exist_ok=True)

    # Convert .csv rows into dictionary
    grouped = rle_pd.groupby('ImageId')[' EncodedPixels'].apply(list).to_dict()

    for i in tqdm(range(len(images)), desc="Preprocessing Mode"):
        # Get image title
        image_id = images[i].stem

        # .dcm -> numpy 
        image = dcmread(images[i]).pixel_array

        # Combine all RLEs into one mask
        shape = image.shape
        mask = np.zeros(shape, dtype=np.uint8)
        for rle in grouped[image_id]:
            mask = np.add(mask, rle2mask(rle, shape[0], shape[1]).astype(np.uint8))

        # Ensure that all pixel values that are between 0-255 are either 0 or 255
        ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

        # Resize images from 1024x1024 -> 256x256
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        # Convert to Image
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if grouped[image_id] == [" -1"]:
            # Save Image as .png
            image.save(f'{output_dir}/normal/image/{image_id}.png')
            mask.save(f'{output_dir}/normal/mask/{image_id}_mask.png')

        else:
            image.save(f'{output_dir}/pneumothorax/image/{image_id}.png')
            mask.save(f'{output_dir}/pneumothorax/mask/{image_id}_mask.png')