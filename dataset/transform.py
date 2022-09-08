from torchvision import transforms
import albumentations as A

def create_transform_resize(size):
    return A.Compose([
        A.Resize(size,size)
    ])