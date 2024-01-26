import sys
import warnings
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append("../")
warnings.filterwarnings("ignore")

def load_and_process_dataset(dataset_name, root_dir, is_train):
    # Load the dataset
    if is_train:
        split = 'train'
    else:
        split = 'val'

    if dataset_name != "Flowers102":
        dataset = datasets.__dict__[dataset_name](root=root_dir, train=is_train, download=True, transform=transforms)

    if dataset_name == "Flowers102":
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

    # CIFAR-10
    cifar10_X_train, cifar10_y_train_true = load_and_process_dataset("CIFAR10", "./data", True)
    cifar10_X_test, cifar10_y_test_true = load_and_process_dataset("CIFAR10", "./data", False)

    np.save('./embedding_data/cifar10_dinov2B_X_train.npy', cifar10_X_train)
    np.save('./embedding_data/cifar10_dinov2B_y_train.npy', cifar10_y_train_true)
    np.save('./embedding_data/cifar10_dinov2B_X_test.npy', cifar10_X_test)
    np.save('./embedding_data/cifar10_dinov2B_y_test.npy', cifar10_y_test_true)

    # CIFAR-100
    cifar100_X_train, cifar100_y_train_true = load_and_process_dataset("CIFAR100", "./data", True)
    cifar100_X_test, cifar100_y_test_true = load_and_process_dataset("CIFAR100", "./data", False)

    np.save('./embedding_data/cifar100_dinov2B_X_train.npy', cifar100_X_train)
    np.save('./embedding_data/cifar100_dinov2B_y_train.npy', cifar100_y_train_true)
    np.save('./embedding_data/cifar100_dinov2B_X_test.npy', cifar100_X_test)
    np.save('./embedding_data/cifar100_dinov2B_y_test.npy', cifar100_y_test_true)

    # Flowers-102
    flowers102_X_train, flowers102_y_train_true = load_and_process_dataset("Flowers102", "./data", 102, True)
    flowers102_X_test, flowers102_y_test_true = load_and_process_dataset("Flowers102", "./data", 102, False)

    np.save('./embedding_data/flowers102_dinov2B_X_train.npy', flowers102_X_train)
    np.save('./embedding_data/flowers102_dinov2B_y_train.npy', flowers102_y_train_true)
    np.save('./embedding_data/flowers102_dinov2B_X_test.npy', flowers102_X_test)
    np.save('./embedding_data/flowers102_dinov2B_y_test.npy', flowers102_y_test_true)
