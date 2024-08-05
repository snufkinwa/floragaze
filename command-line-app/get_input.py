import argparse

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a neural network on flower images')
    
    parser.add_argument('--data_directory', type=str, default='flowers', help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'vgg16'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', nargs=2, type=int, default=[1024, 512], help='Sizes of hidden layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    return parser.parse_args()