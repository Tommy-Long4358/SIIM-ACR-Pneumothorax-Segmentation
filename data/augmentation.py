import albumentations as A
from albumentations.pytorch import ToTensorV2

transform_pneumothorax_image = A.Compose([
    A.Rotate(limit=10, p=1),
    A.GaussianBlur(blur_limit=(3, 5), p=1),
    A.Resize(128, 128),
    ToTensorV2()
])

basic_transform = A.Compose([
    A.Resize(128, 128),
    ToTensorV2()
])