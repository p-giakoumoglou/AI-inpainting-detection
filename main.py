# -*- coding: utf-8 -*-


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from tqdm import tqdm

from iidnet import IID_Net, load_model, IID_Model
from utils import evaluate_model_performance, calculate_pixel_image_metrics


class InpaintingDataset(Dataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for method_dir in os.listdir(self.root_dir):
            method_path = os.path.join(self.root_dir, method_dir)
            if os.path.isdir(method_path):
                for file in os.listdir(method_path):
                    if not file.endswith('_mask.png'):
                        img_path = os.path.join(method_path, file)
                        mask_path = os.path.splitext(img_path)[0] + '_mask.png'
                        if not os.path.exists(mask_path):  # Check if the mask file exists
                            mask_path = None  # Use None to indicate no mask file
                        samples.append((img_path, mask_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')  # Load mask as grayscale if it exists
        else:
            # If mask does not exist, create a dummy mask filled with zeros
            image_size = image.size  # Assuming mask size is same as image size
            mask = Image.new('L', image_size, 0)  # Create a black mask
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert PIL mask to tensor manually if no mask_transform provided
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.  # Normalize to [0, 1]
        
        return image, mask, img_path



def calculate_iou(true_mask, pred_mask):
    # Calculate Intersection and Union
    intersection = (true_mask & pred_mask).sum()
    union = (true_mask | pred_mask).sum()
    # Avoid division by zero
    iou = (intersection / union) if union != 0 else 0
    return iou.item()



#model = IID_Net().cuda()
#load_model(model, 'weights/IID_weights.pth')  

model = IID_Model().cuda()
model.load('')
model.eval()  

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Appropriate for 3-channel images
])

class Binarize(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, tensor):
        # Binarize the tensor: 1 where tensor > threshold, 0 otherwise
        return (tensor > self.threshold).float()

# Then, add this custom transform to your mask transformations
mask_transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize(threshold=0.5)  # Adjust threshold as necessary
])

dataset = InpaintingDataset(root_dir='DiverseInpaintingDataset', 
                            image_transform=image_transform, 
                            mask_transform=mask_transform)
#also cant use num_workers for some reason...
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)


df = evaluate_model_performance(model, dataloader, 'image_metrics.csv')
image_metrics_df = calculate_pixel_image_metrics('image_metrics.csv', 'average_metrics.csv')