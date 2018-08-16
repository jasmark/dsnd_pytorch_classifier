import argparse
import json
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def train_args():

    parser = argparse.ArgumentParser(description='DSND Deep Learning Project')

    parser.add_argument('input_image', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--category_names', type=str)    
    parser.add_argument('--gpu', action="store_true")    

    return parser.parse_args()


def load_checkpoint(checkpoint):
    # load the checkpoint file
    checkpoint = torch.load(checkpoint)

    # restore the model, classifier, and optimizer
    model = eval("models." + checkpoint['pretrained_model'] + '(pretrained=True)')
    classifier = nn.Sequential(checkpoint['classifier_layers'])
    classifier.load_state_dict(checkpoint['classifier_state_dict']),
    model.classifier = classifier
    
    #optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    class_index = {v: k for k, v in checkpoint['class_index'].items()}
    

    return model, class_index

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    return(img)  
    

def predict(image_path, model, category_names, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = torch.FloatTensor(process_image(image_path))
    image = image.unsqueeze_(0)

    model.eval()
    model.to('cpu')

    with torch.no_grad():
        output = model.forward(image)
    
    ps = F.softmax(output, dim=1)
    
    # return predictions and probabilities from model
    probs, preds = torch.topk(ps, topk)

    # translate category names and flatten to array
    preds = preds.data.cpu().numpy().flatten()
    probs = probs.data.cpu().numpy().flatten()

    category_names = get_category_names(category_names)
    preds = [category_names[y] for y in [class_index[x] for x in preds]]

    return preds, probs

def get_category_names(classes):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
 

args = train_args()
model, class_index = load_checkpoint('checkpoint.pth')
probs, preds = predict(args.input_image, model, args.category_names, args.top_k)

print('Chances are this is a: \n')

for i in range(0, len(preds)):
    print(preds[i], probs[i])
    
    