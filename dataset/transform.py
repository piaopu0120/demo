from torchvision import transforms
import albumentations as A

def create_transform_resize(size):
#     return transforms.Compose([
#     transforms.ToPILImage(mode=None),
#     transforms.Resize((size,size)),
#     transforms.autoaugment.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
#     transforms.ToTensor(),
#  ])
    return A.Compose([
        A.Resize(size,size)
    ])