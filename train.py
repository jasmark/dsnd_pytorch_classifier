import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def train_args():

    parser = argparse.ArgumentParser(description='DSND Deep Learning Project')

    parser.add_argument('data_dir', type=str)
    parser.add_argument('--arch', type=str, default="vgg11")
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu', action="store_true")    
    parser.add_argument('--hidden_units', type=int, default=512)

    return parser.parse_args()

def preprocess_data(data_directory):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_val_transforms = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
    
    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return trainloader, validloader, train_data.class_to_idx

def build_model(arch, hidden_units, learning_rate):
    model = model = eval("models." + arch + '(pretrained=True)')

    # freeze params on model
    for param in model.parameters():
        param.requires_grad = False

    # create classifier and connect to model, output 102 classes to match number of flower classes
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p=0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def check_accuracy_on_validation(validloader): 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the %d validation images: %d %%' % (total, (100 * correct / total)))
    return None

def train_model(model, criterion, optimizer, epochs, trainloader, validloader, gpu):
    
    epochs = epochs
    
    device = torch.device('cuda:0' if (torch.cuda.is_available() and gpu) else 'cpu')
    model.to(device)
    
    print_every = 10
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0
                
        check_accuracy_on_validation(validloader)
    return None

def save_checkpoint(save_dir, model, epochs, arch, optimizer, class_index):
    checkpoint = {'number_of_epochs': epochs,
                  'pretrained_model': arch,
                  'classifier_layers': model.classifier._modules,
                  'classifier_state_dict': model.classifier.state_dict(),
                  'class_index': class_index,
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, save_dir + '/' + 'checkpoint.pth')
    return None

args = train_args()

trainloader, validloader, class_index = preprocess_data(args.data_dir)

model, criterion, optimizer = build_model(args.arch, args.hidden_units, args.learning_rate)

train_model(model = model, optimizer = optimizer, criterion = criterion,
            epochs = args.epochs, trainloader = trainloader, validloader = validloader, gpu = args.gpu)

save_checkpoint(save_dir = args.save_dir, model = model, epochs = args.epochs, 
                arch = args.arch, optimizer = optimizer, class_index = class_index)

