from dataloaders import PennFudanDataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader



def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
data_loader = DataLoader(
    dataset, batch_size=2, shuffle=True
)

for batch in data_loader:
    print(batch)