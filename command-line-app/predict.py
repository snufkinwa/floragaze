import argparse
import torch
from model import load_checkpoint
from utils import process_image, load_category_names

def predict(image_path, model, topk=5, device='cpu', idx_to_class=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''

    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        output = model.forward(img)
        
    probs = torch.exp(output).data.topk(topk)[0].tolist()[0]
    indices = torch.exp(output).data.topk(topk)[1].tolist()[0]
    
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def get_predict_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def main():
    args = get_predict_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  
    
    model, _, class_to_idx = load_checkpoint(args.checkpoint)
    model.to(device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    cat_to_name = load_category_names(args.category_names)
 
    probs, classes = predict(args.image_path, model, args.top_k, device, idx_to_class)
 
    print("\nTop {} Classes:".format(args.top_k))
    for i in range(args.top_k):
        print(f"{cat_to_name[classes[i]]}: {probs[i]:.3f}")

if __name__ == '__main__':
    main()