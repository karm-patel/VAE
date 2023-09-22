import os
import zipfile
from natsort import natsorted
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_DIM = 784
Z_DIM = 20
H_DIM = 200
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
DATA_PATH = os.path.join(DATA_DIR_PATH, 'afhq')
DATA_URL = 'https://www.kaggle.com/datasets/andrewmvd/animal-faces/download?datasetVersionNumber=1'


# Prepare custom Dataloader class
class AnimalfaceDataset(Dataset):
    def __init__(self, type='train', transform=None) -> None:
        # self.root_dir specifies weather you are at afhq/train or afhq/val directory
        self.root_dir = os.path.join(DATA_PATH, type)
        assert os.path.exists(self.root_dir), "Check for the dataset, it is not where it should be. If not present, you can download it by clicking above DATA_URL"
        subdir = os.listdir(self.root_dir)
        image_names = []
        for category in tqdm(subdir):
            subdir_path = os.path.join(self.root_dir, category)
            image_names+=[os.path.join(category,i) for i in os.listdir(subdir_path)]
        self.image_names = natsorted(image_names)
        self.transform = transform
    
    def __getitem__(self, idx):
        # Get the path to the image 
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.image_names)
    
    def __getnames__(self):
        return self.image_names

transform = transforms.Compose([
                    transforms.Resize(size = (128,128)),
                    transforms.ToTensor()
                    ])
train_data = AnimalfaceDataset(transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data = AnimalfaceDataset(type='val', transform=transform)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

'''
PLOT AN IMAGE
print(next(iter(train_loader))[0].shape)
plt.imshow(next(iter(train_loader))[0].permute(1, 2, 0))
plt.savefig("test_im.png")
'''

'''
# Train
model = VAE(X_DIM, H_DIM, Z_DIM)
optim = torch.optim.Adam(model.parameters(), lr=LR_RATE)  # , weight_decay=1e-5
for i in range(NUM_EPOCHS):
    for itr,x in enumerate(train_loader):
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum() + model.enc.kl

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()


# # for personal testing
# if __name__=='__main__':
#     obj = AnimalfaceDataset(type='val')
#     print(obj.__len__())
'''