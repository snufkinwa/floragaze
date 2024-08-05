import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from model import create_model, save_checkpoint
from utils import load_data, download_and_extract_data, split_data
from get_input import get_input_args
from tqdm import tqdm
import os

def train_model(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    
    train_dir = os.path.join(data_directory, 'train')
    valid_dir = os.path.join(data_directory, 'valid')
    test_dir = os.path.join(data_directory, 'test')
    
    if not (os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir)):
        print("Data not found or not properly organized. Downloading and preparing data...")
        download_and_extract_data(data_directory)
        split_data(data_directory)
    else:
        print("Data already present and organized. Skipping download and preparation.")
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloaders, class_to_idx = load_data(data_directory)
    if dataloaders is None or class_to_idx is None:
        print("Error loading data. Exiting.")
        return

    model = create_model(arch, hidden_units, len(class_to_idx))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    if arch == 'resnet50':
        optimizer = optim.Adam([
            {'params': model.fc.parameters(), 'lr': learning_rate},
            {'params': model.layer4.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.layer3.parameters(), 'lr': learning_rate * 0.01}
        ])
    elif arch == 'vgg16':
        optimizer = optim.Adam([
            {'params': model.classifier.parameters(), 'lr': learning_rate},
            {'params': model.features[-4:].parameters(), 'lr': learning_rate * 0.1},
            {'params': model.features[-8:-4].parameters(), 'lr': learning_rate * 0.01}
        ])
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        if arch == 'resnet50':
            if epoch == 10:
                for param in model.layer4.parameters():
                    param.requires_grad = True
            if epoch == 20:
                for param in model.layer3.parameters():
                    param.requires_grad = True
        elif arch == 'vgg16':
            if epoch == 10:
                for param in model.features[-4:].parameters():
                    param.requires_grad = True
            if epoch == 20:
                for param in model.features[-8:-4].parameters():
                    param.requires_grad = True
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {epoch_loss:.4f}, Validation loss: {epoch_valid_loss:.4f}")
        print(f"Validation accuracy: {epoch_accuracy:.4f}")
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            save_checkpoint(model, save_dir, arch, hidden_units, epoch, class_to_idx)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        scheduler.step()
    
    print("Training complete")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    args = get_input_args()
    train_model(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)