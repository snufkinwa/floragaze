import torch
from torch import nn
from torchvision import models
import torch
from torch import nn
from torchvision import models

def create_model(arch='resnet50', hidden_units=[1024, 512], output_size=102, weights='IMAGENET1K_V2'):
    if arch == 'resnet50':
        if weights == 'IMAGENET1K_V2':
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            raise ValueError(f"Weight {weights} is not supported.")
        model = models.resnet50(weights=weights)
       
        for param in model.parameters():
            param.requires_grad = False
       
        num_ftrs = model.fc.in_features
    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
      
        num_ftrs = model.classifier[0].in_features
    else:
        raise ValueError(f"Model architecture {arch} is not supported.")

    classifier = nn.Sequential(
        nn.Linear(num_ftrs, hidden_units[0]),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units[0], hidden_units[1]),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units[1], output_size),
        nn.LogSoftmax(dim=1)
    )

    if arch == 'resnet50':
        model.fc = classifier
    elif arch == 'vgg16':
        model.classifier = classifier

    return model

def save_checkpoint(model, save_dir, arch, hidden_units, epochs, class_to_idx):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }
    torch.save(checkpoint, f'{save_dir}/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = create_model(arch=checkpoint['arch'], 
                         hidden_units=checkpoint['hidden_units'],
                         output_size=len(checkpoint['class_to_idx']))
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    return model, epochs, class_to_idx
