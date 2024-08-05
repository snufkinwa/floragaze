# -*- coding: utf-8 -*-
"""Image Classifier Project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hCgm8K5bFMvfP4_fHlBocSlq4TbEPI8f

# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.
"""

# Commented out IPython magic to ensure Python compatibility.
# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import copy
import os
import tarfile
import urllib.request
import scipy.io
import shutil
from scipy.io import loadmat
from torchvision.models import ResNet50_Weights

"""## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.

"""

# Define the directories
data_dir = 'flowers'
os.makedirs(data_dir, exist_ok=True)

# Download the dataset
flowers_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
splits_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

flowers_path = os.path.join(data_dir, "102flowers.tgz")
labels_path = os.path.join(data_dir, "imagelabels.mat")
splits_path = os.path.join(data_dir, "setid.mat")

urllib.request.urlretrieve(flowers_url, flowers_path)
urllib.request.urlretrieve(labels_url, labels_path)
urllib.request.urlretrieve(splits_url, splits_path)

# Extract the dataset
with tarfile.open(flowers_path, 'r:gz') as tar_ref:
    tar_ref.extractall(data_dir)

print("Dataset and labels downloaded and extracted.")

# Load the splits
setid = loadmat(splits_path)
train_ids = setid['trnid'][0]
valid_ids = setid['valid'][0]
test_ids = setid['tstid'][0]

# Load the image labels
image_labels = loadmat(labels_path)['labels'][0]

# Define directories for splits
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to move images to corresponding directories
def move_images(image_ids, split_dir):
    for image_id in image_ids:
        label = image_labels[image_id - 1]
        label_dir = os.path.join(split_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(data_dir, 'jpg', f'image_{image_id:05d}.jpg')
        dst = os.path.join(label_dir, f'image_{image_id:05d}.jpg')
        shutil.move(src, dst)

# Move images to train, valid, and test directories
move_images(train_ids, train_dir)
move_images(valid_ids, valid_dir)
move_images(test_ids, test_dir)

print("Images have been organized into train, valid, and test directories.")

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Get the num of classes from the directory
num_train_classes = len(os.listdir(train_dir))
num_valid_classes = len(os.listdir(valid_dir))
num_test_classes = len(os.listdir(test_dir))

print(num_train_classes)
print(num_valid_classes)
print(num_test_classes)

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
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

# TODO: Load the datasets with ImageFolder
image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
# train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
# valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
# test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

batch_size = 32


# TODO: Using the image datasets and the trainforms, define the dataloaders
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
    }

"""### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.
"""

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    print(cat_to_name)

"""# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.
"""

# TODO: Build and train your network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 102),
    nn.LogSoftmax(dim=1)
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': model.layer3.parameters(), 'lr': 0.00001}
])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    if epoch == 10:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if epoch == 20:
        for param in model.layer3.parameters():
            param.requires_grad = True

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    valid_loss = 0.0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloaders['train'])
    epoch_valid_loss = valid_loss / len(dataloaders['valid'])
    epoch_accuracy = accuracy.double() / len(dataloaders['valid'].dataset)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train loss: {epoch_loss:.4f}, Validation loss: {epoch_valid_loss:.4f}")
    print(f"Validation accuracy: {epoch_accuracy:.4f}")

    scheduler.step()

print("Training complete")

"""## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.
"""

# TODO: Do validation on the test set
def validate_model(model, dataloaders, criterion, device):
    test_loss = 0
    test_accuracy = 0
    total = 0

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Calculate loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            test_accuracy += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_test_loss = test_loss / len(dataloaders['test'].dataset)
    avg_test_accuracy = test_accuracy / total

    print(f"\nTest Results:")
    print(f"  Loss: {avg_test_loss:.4f}")
    print(f"  Accuracy: {avg_test_accuracy:.4f}")

    return avg_test_loss, avg_test_accuracy


test_loss, test_accuracy = validate_model(model, dataloaders, criterion, device)

"""## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.
"""

# TODO: Save the checkpoint

def save_checkpoint(model, optimizer, class_to_idx, epochs, filepath='checkpoint.pth'):
    checkpoint = {
        'arch': 'resnet50',
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'classifier': model.fc,
        'epochs': epochs
    }
    torch.save(checkpoint, filepath)

model.class_to_idx = image_datasets['train'].class_to_idx


save_checkpoint(model, optimizer, model.class_to_idx, num_epochs)

"""## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.
"""

def load_checkpoint(filepath='checkpoint.pth', device='cpu'):
    try:
        checkpoint = torch.load(filepath, map_location=device)

        if checkpoint['arch'] == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Model architecture {checkpoint['arch']} is not supported.")

        model.fc = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']


        optimizer = optim.Adam(model.fc.parameters())

        # We can still the optimizer state, but caused error in initial execution
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model.to(device), optimizer

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None, None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, optimizer = load_checkpoint('checkpoint.pth', device)

if model is not None:
    print("Checkpoint loaded successfully.")
else:
    print("Failed to load checkpoint.")

"""# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network.

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training.

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.
"""

def process_image(image_path):
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

"""To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions)."""

def imshow(image, ax=None, title=None):
    """Imshow for Numpy array."""
    if ax is None:
        fig, ax = plt.subplots()

    if image.shape[0] == 3:
        image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.imshow(image)

    if title:
        ax.set_title(title)

    return ax


image_path = '/content/flowers/valid/1/image_06738.jpg'
processed_image = process_image(image_path)
plt.figure(figsize=(2.24, 2.24))
ax = imshow(processed_image)
plt.show()

"""## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```
"""

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)

    image = torch.from_numpy(image).type(torch.FloatTensor)

    image = image.unsqueeze(0)

    device = next(model.parameters()).device
    image = image.to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)

    ps = torch.exp(output)

    top_p, top_class = ps.topk(topk, dim=1)

    top_p = top_p.cpu()
    top_class = top_class.cpu()

    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]

    return top_p, top_class

image_path = '/content/flowers/valid/102/image_08008.jpg'
probs, classes = predict(image_path, model)
print(probs)
print(classes)

def display_prediction(image_path, model):
    probs, classes = predict(image_path, model)

    print(f"Image: {image_path}")
    print("\nTop 5 predictions:")

    for prob, class_label in zip(probs, classes):
        percentage = prob * 100
        flower_name = cat_to_name.get(class_label, f"Unknown (Class {class_label})")
        print(f"{flower_name}: {percentage:.2f}%")

# Now let's use this function
image_path = '/content/flowers/valid/102/image_08008.jpg'
display_prediction(image_path, model)

"""## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.
"""

# TODO: Display an image along with the top 5 classes

def plot_sanity_check(image_path, model, cat_to_name, topk=5):
    ''' Plot the input image along with the top K class probabilities
    '''
    probs, classes = predict(image_path, model, topk=topk)

    flower_names = [cat_to_name[cls] for cls in classes]

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), nrows=2)

    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(flower_names[0])

    y_pos = np.arange(len(flower_names))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flower_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top {} Predictions'.format(topk))

    plt.tight_layout()
    plt.show()

image_path = '/content/flowers/test/36/image_04353.jpg'
plot_sanity_check(image_path, model, cat_to_name, topk=5)