'''
Assignment 2: Applying Transformations using torchvision.transforms

Problem 1: Load an image using PIL, convert it to a PyTorch tensor, and normalize it using torchvision.transforms.
Normalize with mean=0.5 and std=0.5.

Problem 2: Create a custom transformation that rotates an image by 45 degrees and apply it to an image. Use torchvision.transforms.Compose to chain this custom transformation with a resize transformation that resizes the image to 128x128 pixels.

Problem 3: Use torchvision.transforms to apply the following transformations to an image:

Random horizontal flip.
Random crop of size 100x100.
Convert the image to grayscale.
'''

from PIL import Image
import torchvision.transforms as transforms
import torch

# Problem 1: Load an image, convert to tensor, and normalize
def problem1(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image)
    print("Problem 1 - Tensor shape:", image_tensor.shape)
    print("Problem 1 - Tensor values:", image_tensor)
    return image_tensor

# Problem 2: Custom rotate transformation and resize
class RotateTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return x.rotate(self.angle)

def problem2(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        RotateTransform(45),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    transformed_image = transform(image)
    print("Problem 2 - Transformed Tensor shape:", transformed_image.shape)
    print("Problem 2 - Transformed Tensor values:", transformed_image)
    return transformed_image

# Problem 3: Random horizontal flip, random crop, and grayscale conversion
def problem3(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((100, 100)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    transformed_image = transform(image)
    print("Problem 3 - Transformed Tensor shape:", transformed_image.shape)
    print("Problem 3 - Transformed Tensor values:", transformed_image)
    return transformed_image

if __name__ == '__main__':
    image_path = 'flower.jpg'  # Replace with the actual path to your image
    problem1(image_path)
    problem2(image_path)
    problem3(image_path)


