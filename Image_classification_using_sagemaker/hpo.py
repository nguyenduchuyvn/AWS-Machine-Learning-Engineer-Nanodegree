#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import argparse
import os
import sys
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, loss_criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Testing started.")
    test_loss = 0
    correct = 0
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            
            output = model(data)
            test_loss += loss_criterion(output, target).item()  # sum up batch loss
            _, pred = torch.max(output, 1)           
            correct += torch.sum(pred==target.data).item()

    test_loss /= len(test_loader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {correct / len(test_loader.dataset)}")
    logger.info(f"Test Loss: {test_loss}")

def train(model, train_loader, valid_loader, loss_criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Start training model.")
    loss_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0
        model.train()
        model.to(device)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch}: Train loss = {train_loss:.4f}")

        val_loss = 0
        model.eval()

        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)

                outputs = model(data)

                loss = loss_criterion(outputs, target)
                val_loss += loss.item()

            val_loss /= len(valid_loader.dataset)
            print(f"Epoch {epoch}: Val loss = {val_loss:.4f}")

    logger.info("Training completed.")
    
def net(num_classes):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    logger.info("Start creating model.")
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    logger.info("Model has created.")
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def create_transform(split, image_size):
    logger.info("Start Transformation pipeline ")

    pretrained_size = image_size

    if split == "train":
        train_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])

        logger.info("Transformation pipeline has completed")
        return train_transforms

    elif split == "valid":
        valid_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)),  
            transforms.ToTensor(),
        ])

        logger.info("Transformation pipeline has completed")
        return valid_transforms

    elif split == "test":
        test_transforms = transforms.Compose([
            transforms.Resize((image_size,image_size)),     
            transforms.ToTensor(),

        ])
        logger.info("Transformation pipeline has completed")
        return test_transforms
    
    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=net(args.num_classes)
    '''
    TODO: Create your loss and optimizer
    '''
    optimizer = optim.Adadelta(model.parameters(), lr= args.lr)
    loss_criterion = nn.CrossEntropyLoss()

    
    train_dir = os.path.join(args.data_dir, 'train')
    valid_dir = os.path.join(args.data_dir, 'valid')
    test_dir = os.path.join(args.data_dir, 'test')

    transform_train = create_transform("train", args.image_size)
    transform_test = create_transform("test", args.image_size)
    transform_valid = create_transform("valid", args.image_size)

    train_dataset = datasets.ImageFolder(root= train_dir, transform= transform_train)
    valid_dataset = datasets.ImageFolder(root= valid_dir, transform= transform_valid)
    test_dataset = datasets.ImageFolder(root= test_dir, transform= transform_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= False)
    valid_loader = DataLoader(valid_dataset,  batch_size=args.batch_size, shuffle=False)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model.cpu().state_dict(), 
                os.path.join(args.model_path,"model.pth"))
    logger.info("Model weights saved.")



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument("--num_classes", type=int, default= 133)
    parser.add_argument("--epochs", type=int, default= 5)
    parser.add_argument("--lr", type=float, default= 0.01)
    parser.add_argument("--batch_size", type=int, default= 16)
    parser.add_argument("--image_size", type=int, default= 224)
    parser.add_argument("--data_dir",  type=str, default= os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model_path",  type=str, default= os.environ["SM_MODEL_DIR"])
    args=parser.parse_args()
    
    main(args)
