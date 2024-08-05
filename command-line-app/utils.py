import os
import tarfile
import urllib.request
import shutil
from scipy.io import loadmat
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch
import json


def download_and_extract_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    flowers_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    splits_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    
    flowers_path = os.path.join(data_dir, "102flowers.tgz")
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    splits_path = os.path.join(data_dir, "setid.mat")
    
    if not os.path.exists(flowers_path):
        print(f"Downloading {flowers_url} ...")
        urllib.request.urlretrieve(flowers_url, flowers_path)
    if not os.path.exists(labels_path):
        print(f"Downloading {labels_url} ...")
        urllib.request.urlretrieve(labels_url, labels_path)
    if not os.path.exists(splits_path):
        print(f"Downloading {splits_url} ...")
        urllib.request.urlretrieve(splits_url, splits_path)
    
    if not os.path.exists(os.path.join(data_dir, 'jpg')):
        print(f"Extracting {flowers_path} ...")
        with tarfile.open(flowers_path, 'r:gz') as tar_ref:
            tar_ref.extractall(data_dir)

    print("Extraction complete. Contents of the data directory:")
    for item in os.listdir(data_dir):
            print(item)

    else:
        print("Data already extracted.")

def split_data(data_dir):
    labels_path = os.path.join(data_dir, "imagelabels.mat")
    splits_path = os.path.join(data_dir, "setid.mat")
    jpg_dir = os.path.join(data_dir, 'jpg')

    if not os.path.exists(jpg_dir):
        print(f"Error: 'jpg' directory not found in {data_dir}")
        print("Current directory structure:")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{subindent}{f}")
        return

    setid = loadmat(splits_path)
    train_ids = setid['trnid'][0]
    valid_ids = setid['valid'][0]
    test_ids = setid['tstid'][0]

    image_labels = loadmat(labels_path)['labels'][0]

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    def move_images(image_ids, split_dir):
        for image_id in image_ids:
            label = image_labels[image_id - 1]
            label_dir = os.path.join(split_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            src = os.path.join(jpg_dir, f'image_{image_id:05d}.jpg')
            dst = os.path.join(label_dir, f'image_{image_id:05d}.jpg')
            if not os.path.exists(src):
                print(f"Warning: Source file not found: {src}")
                continue
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"Error moving file {src} to {dst}: {str(e)}")

    print("Moving training images...")
    move_images(train_ids, train_dir)
    print("Moving validation images...")
    move_images(valid_ids, valid_dir)
    print("Moving test images...")
    move_images(test_ids, test_dir)

    print("Data split complete.")
    
def load_category_names(json_file):
    """
    Load category names from a JSON file
    """
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_data(data_directory, batch_size=32):
    train_dir = os.path.join(data_directory, 'train')
    valid_dir = os.path.join(data_directory, 'valid')
    test_dir = os.path.join(data_directory, 'test')
    
    if not all(os.path.exists(d) for d in [train_dir, valid_dir, test_dir]):
        print(f"Error: One or more data directories not found in {data_directory}")
        return None, None
    
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    # Define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size)
    }
    
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a NumPy array
    """
    image = Image.open(image_path)
    
    if image.size[0] < image.size[1]:
        image.thumbnail((256, 256 * image.size[1] // image.size[0]))
    else:
        image.thumbnail((256 * image.size[0] // image.size[1], 256))
    
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
