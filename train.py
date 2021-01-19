import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


def arg_parser():

    parser = argparse.ArgumentParser()

                      
    parser.add_argument('--data_dir',
                        type=str, 
                        default=data_dir,
                        help = 'Dataset path')
    
    parser.add_argument('--save_dir', 
                        type=str, 
                        default='ch.pth',
                        help='save directory for checkpoint. If not specified then model will be lost.')

    parser.add_argument('--arch', 
                        type=str, 
                        default='vgg19',
                        help='Choose architecture vgg19 or vgg16')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='learning rate as float',
                        default=0.001)
   
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Number of hidden units',
                        default=1000)
   
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs',
                        default=11)

    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU or CPU',
                        default=True)
    

    parser.add_argument('--num_classes',
                        help='out put classes',
                        default=102)
    
    args = parser.parse_args()
    
    return args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu, args.num_classes



def train_loader(filepath=train_dir, valid_filepath=valid_dir):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(filepath, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = test_loader(valid_filepath)
    return trainloader, train_data, validloader


def test_loader(filepath=test_dir):
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    test_data = datasets.ImageFolder(filepath, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return testloader


def load_model(arch, hidden_units, num_classes):
    
    model = getattr(models, arch)(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(hidden_units, num_classes),
                                     nn.LogSoftmax(dim = 1)
                                    )
    model.classifier = classifier
    model.name = arch 
    
    return model
    
def train(epochs, optimizer, criterion, trainloader, train_data, validloader, model, device, save_dir):
    model.to(device)
    for e in range(epochs):
        train_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            with torch.no_grad():
                model.eval()
                test_loss, accuracy = test_model(model, validloader, criterion, device)
                
            print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {train_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {test_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            model.train()

    if save_dir != None:

        print('\n\n Training is complete saving model...')
        model.class_to_idx = train_data.class_to_idx

        checkpoint = {'architecture': model.name,
                      'classifier': model.classifier,
                      'class_to_idx': model.class_to_idx,
                      'state_dict': model.state_dict()}

        torch.save(checkpoint, save_dir)






def test_model(model, testloader, criterion, device):
    model.to(device)
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        test_loss += criterion(logps, labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, accuracy

    
    
    
    
    
    
def test_model(model, loader, criterion, device):
    model.to(device)
    accuracy = 0
    test_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        test_loss += criterion(logps, labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    return test_loss, accuracy




def main():
    
    data_directory, save_directory, arch, learning_rate, hidden_units, epochs, gpu, num_classes = arg_parser()
    if gpu==True:
        device = 'cuda'
    else:
        device = 'cpu'
    
    trainloader, train_data, validloader = train_loader()
    
    model = load_model(arch, hidden_units, num_classes)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    train(epochs, optimizer, criterion, trainloader, train_data, validloader, model, device, save_directory) 
    
    with torch.no_grad():
        model.eval()
        testloader = test_loader()
        _, accuracy = test_model(model, testloader, criterion, device)
        print('Accuracy:  {}'.format(accuracy/len(testloader)))
              
              
              
if __name__ == '__main__':
    
    main()