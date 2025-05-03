import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Define custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.classes = ['dogs']

        for label in self.classes:
            for img_name in os.listdir(os.path.join(img_dir, label)):
                self.img_labels.append((os.path.join(img_dir, label, img_name), self.classes.index(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and dataloader
image_dataset = CustomImageDataset(img_dir='dataset', transform=transform)

image_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True, num_workers=0)

# Iterate through the dataloader and display information
for batch_idx, (images, labels) in enumerate(image_dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Images Shape: {images.shape}")
    print(f"Labels: {labels}")
    break  # Just process the first batch for this example