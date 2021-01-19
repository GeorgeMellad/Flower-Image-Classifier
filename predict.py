import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, default = 'flowers/test/102/image_08012.jpg', help = 'image path')
    parser.add_argument('--checkpoint_path', type = str, default = 'ch.pth', help = 'trained model path')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'File .json for the categories')
    parser.add_argument('--gpu', action = "store_true", default = True, help = 'Use GPU if available')

    args = parser.parse_args()
    
    return args.image_path, args.checkpoint_path, args.top_k, args.category_names, args.gpu


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    
    model = getattr(models, checkpoint['architecture'])(pretrained = True)
    
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    
    im = Image.open(image_path)
    size = 256,256
    im = im.resize(size)
    left_margin = (im.width-224)/2
    bottom_margin = (im.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    im = im.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    np_image = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk, device):
    
    model.to(device)

    model.eval()

    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)
    logps = model.forward(torch_image)
    ps = torch.exp(logps)
    top_p, top_c = ps.topk(topk, dim=1)
    class_to_idx_inverted = {model.class_to_idx[c]: c for c in model.class_to_idx}
    top_classes = []
    
    for label in top_c.cpu().detach().numpy()[0]:
        top_classes.append(class_to_idx_inverted[label])
    
    top_p = top_p.cpu().detach().numpy()[0]
    
    return top_p, top_classes

def get_name_categories(category_names):
    
    with open(category_names, 'r') as f:
        category_label_to_name = json.load(f)
    
    return category_label_to_name

def main():
    
    image_path, checkpoint_path, top_k, category_names, gpu = arg_parser()
    
    model = load_checkpoint(checkpoint_path)
    
    np_image = process_image(image_path)
    
    category_label_to_name = get_name_categories(category_names)
    
    if gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'
    
    top_probs, top_classes = predict(image_path, model, top_k, device)
    
    print('Probabilities: ', top_probs)
    print('Categories:    ', [category_label_to_name[c] for c in top_classes])
    

if __name__ == '__main__':
   
    main()
    
    
    
    
    
    