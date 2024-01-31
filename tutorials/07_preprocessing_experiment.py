import sys
import warnings
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("/mnt/stud/home/jcheng/scikit-activeml/")
warnings.filterwarnings("ignore")

import argparse

def parse_argument():
    parser = argparse.ArgumentParser(description='Preprocessing Data with DinoV2')
    parser.add_argument('dataset', type=str, help='name of dataset')
    return parser

def load_and_process_dataset(dataset_name, root_dir, is_train):
    # Load the dataset
    if is_train:
        split = 'train'
    else:
        if dataset_name == "STL10":
            split = 'test'
        else:
            split = 'val'

    if dataset_name != "Flowers102" and dataset_name != "STL10":
        dataset = datasets.__dict__[dataset_name](root=root_dir, train=is_train, download=True, transform=transforms)
    else:
        dataset = datasets.__dict__[dataset_name](root=root_dir, split=split, download=True, transform=transforms)
        
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)

    embedding_list = []
    label_list = []

    # Compute the embedding
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{dataset_name.capitalize()} {split}"):
            image, label = data
            embeddings = dinov2_vitb14(image.to(device)).cpu()
            embedding_list.append(embeddings)
            label_list.append(label)

        # Concatenate embeddings and labels
        X = torch.cat(embedding_list, dim=0).numpy()
        y_true = torch.cat(label_list, dim=0).numpy()

    return X, y_true

if __name__ == "__main__":
    parser = parse_argument()
    args = parser.parse_args()
    dataset_name = args.dataset
    
    # 1. Transformation
    transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    batch_size = 4

    dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_vitb14.to(device)

    X_train, y_train_true = load_and_process_dataset(dataset_name, "./data", True)
    X_test, y_test_true = load_and_process_dataset(dataset_name, "./data", False)
    file_name = dataset_name.lower()

    np.save(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{file_name}_dinov2B_X_train.npy', X_train)
    np.save(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{file_name}_dinov2B_y_train.npy', y_train_true)
    np.save(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{file_name}_dinov2B_X_test.npy', X_test)
    np.save(f'/mnt/stud/home/jcheng/scikit-activeml/tutorials/embedding_data/{file_name}_dinov2B_y_test.npy', y_test_true)

