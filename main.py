
from tqdm import tqdm
import os
import sys
import pandas as pd
#import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from PIL import Image
#import medmnist
#from medmnist import INFO, Evaluator


sys.path.append(r'A')
sys.path.append(r'B')

import TaskA as A
import TaskB as B


# Check the availablity of Using Cuda to accerate training process
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialization method to create an instance of CustomDataset
class customdataset():
    def __init__(self, csv_file, img_dir, transform=None) -> None:
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

# Get image and label according to the csv file
    def __getitem__(self, idx):
        # Get image
        img_name = self.img_labels.iloc[idx,1]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        # Get label
        label = self.img_labels.iloc[idx,2]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_classes(csv_file):
        data = pd.read_csv(csv_file)
        return sorted(set(data.iloc[:,2]))


while True:
    print('Please choose the dataset you want to use:')
    print('1: PneumoniaMNIST\n2: PathMNIST\n')
    x = input()
    if x == '1' or x.lower() == 'pneumoniamnist':
        data_flag = 'pneumoniamnist'

        NUM_EPOCHS = 30
        BATCH_SIZE = 128
        lr = 0.001

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_dataset = customdataset(csv_file=r'Datasets/pneumoniamnist/train/pneumoniamnist.csv', img_dir=r'Datasets/pneumoniamnist/train/pneumoniamnist', transform=train_data_transform)
        val_dataset = customdataset(csv_file=r'Datasets/pneumoniamnist/validate/pneumoniamnist.csv', img_dir=r'Datasets/pneumoniamnist/validate/pneumoniamnist', transform=data_transform)
        test_dataset = customdataset(csv_file=r'Datasets/pneumoniamnist/test/pneumoniamnist.csv', img_dir=r'Datasets/pneumoniamnist/test/pneumoniamnist', transform=data_transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print('PneumoniaMNIST is ready')
        break

    elif x == '2' or x.lower() == 'pathmnist':
        data_flag = 'pathmnist'
        
        NUM_EPOCHS = 3
        BATCH_SIZE = 128
        lr = 0.001

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(degrees=(-30, 30)),
            #transforms.ColorJitter(brightness=0.2, contrast=0.1,saturation=0.1,hue=0.1),
            #transforms.RandomGrayscale(p=0.025),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_dataset = customdataset(csv_file=r'Datasets/pathmnist/train/pathmnist.csv', img_dir=r'Datasets/pathmnist/train/pathmnist', transform=train_data_transform)
        val_dataset = customdataset(csv_file=r'Datasets/pathmnist/validate/pathmnist.csv', img_dir=r'Datasets/pathmnist/validate/pathmnist', transform=data_transform)
        test_dataset = customdataset(csv_file=r'Datasets/pathmnist/test/pathmnist.csv', img_dir=r'Datasets/pathmnist/test/pathmnist', transform=data_transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print('PathMNIST is ready')
        break 
    else:
        print('Error: Please choose the correct number or dataset name')


# Online dataset can be used instead.
# However, these line should be resized when using online dataset
# Please remove these two lines in the TaskA training module
#targets = targets.unsqueeze(1)
#val_targets = val_targets.unsqueeze(1)
# Please remove this line in TaskA testing module
#targets = targets.unsqueeze(1)
# You should able to run this part after you done the revise
'''
#Dataset and hyperparameter choosing
while True:
    print('Please choose the dataset you want to use:')
    print('1: PneumoniaMNIST\n2: PathMNIST\n')
    x = input()
    
    if x == '1' or x.lower() == 'pneumoniamnist':
        data_flag = 'pneumoniamnist'

        # Hyperparameters setting for pneumoniamnist
        NUM_EPOCHS = 3
        BATCH_SIZE = 128
        lr = 0.001
        
        info = INFO[data_flag]
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        class_names = ['Normal','Pneumonia']
        DataClass = getattr(medmnist, info['python_class'])

        # data augumentation
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        download = True
        train_dataset = DataClass(split='train', transform=train_data_transform, download=download)
        val_dataset = DataClass(split='val',transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)


        # encapsulate data into dataloader form
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print('PneumoniaMNIST is ready')
        break
    elif x == '2' or x.lower() == 'pathmnist':
        data_flag = 'pathmnist'

        #Hyperparameters setting for pathmnist
        NUM_EPOCHS = 30
        BATCH_SIZE = 128
        lr = 0.001

        info = INFO[data_flag]
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        # data augumentation
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        download = True
        train_dataset = DataClass(split='train', transform=train_data_transform, download=download)
        val_dataset = DataClass(split='val',transform=data_transform, download=download)
        test_dataset = DataClass(split='test', transform=data_transform, download=download)


        # encapsulate data into dataloader form
        train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)


        print('PathMNIST is ready')
        break 
    else:
        print('Error: Please choose the correct number or dataset name')
'''

# Action selection
while True:
    print('Please select what action you want to proceed:\n')
    print('1: train\n2: view model performance\n')
    x = input()
    # For training 
    if x == '1' or x.lower() == 'train':
        
        if data_flag == 'pneumoniamnist':
            print('CNN3 LOADING =====================')
            # Load the hyperparameters for training CNN3
            model = A.CNN3()
            model = model.to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(),lr=lr)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            print('CNN3 LOADED ======================')
            print('MODEL VIEWING ========================')
            print(model)
            print('MODEL TRAINING =======================')
            print('Hyperparameters: optimizer=Adam, Lr=0.001, Loss=Binary Cross Entropy Loss, Batch_size=128, Epoch=30, Early stop=yes')
            train_loss_history, train_acc_history, val_loss_history, val_acc_history, epoch_number = A.train_model(model, device, train_loader, val_loader, optimizer, criterion, scheduler, NUM_EPOCHS)
            print('MODEL TRAINING COMPLETED =============')
            print('PLOTING LEARNING CURVES ==============')
            A.curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history, epoch_number)
            print('PLOTING COMPLETED ====================')
            print('TESTING THE MODEL ====================')
            A.test(model, device, test_loader)

        if data_flag == 'pathmnist':
            print('Resnet50 LOADING =====================')
            # Load the hyperparameters for training Resnet50
            model = B.RestNet50()
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(),lr=lr)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
            print('Resnet50 LOADED ======================')
            print('MODEL VIEWING ========================')
            print(model)
            print('MODEL TRAINING =======================')
            print('Hyperparameters: optimizer=Adam, Lr=0.001, Loss=Cross Entropy Loss, Batch_size=128, Epoch=30, Early stop=yes')
            train_loss_history, train_acc_history, val_loss_history, val_acc_history, epoch_number= B.train_model(model, device, train_loader, val_loader, optimizer, criterion, scheduler, NUM_EPOCHS)
            print('MODEL TRAINING COMPLETED =============')
            print('PLOTING LEARNING CURVES ==============')
            B.curve(train_loss_history, train_acc_history, val_loss_history, val_acc_history, epoch_number)
            print('PLOTING COMPLETED ====================')
            print('TESTING THE MODEL ====================')
            B.test(model, device, test_loader)


    elif x == '2' or x.lower() == 'view':
        # For viewing action
        if data_flag == 'pneumoniamnist':
            print('Which model would you like to view in PneumoniaMNIST?\n')
            print('1: CNN3\n2: Resnet18\n3: LeNet\n')
            y = input()
            # CNN3 viewing
            if y == '1' or x.lower() == 'cnn3':
                print('CNN3 MODEL LOADING ========================')
                model = A.CNN3()
                model = model.to(device)
                path = './A/CNN3'
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                print('MODEL LOADED =========================')
                print('MODEL VIEWING ========================')
                print(model)
                print('TESTING THE MODEL ====================')
                A.test(model, device, test_loader)
                break
            # resnet18 viewing
            if y == '2' or x.lower() == 'resnet18':
                print('Resnet18 MODEL LOADING ========================')
                model = A.ResNet18()
                model = model.to(device)
                path = './A/Resnet18'
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                print('MODEL LOADED =========================')
                print('MODEL VIEWING ========================')
                print(model)
                print('TESTING THE MODEL ====================')
                A.test(model, device, test_loader)
                break
            # LeNet viewing
            if y == '3' or x.lower() == 'lenet':
                print('LeNet MODEL LOADING ========================')
                model = A.LeNet()
                model = model.to(device)
                path = './A/LeNet'
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                print('MODEL LOADED =========================')
                print('MODEL VIEWING ========================')
                print(model)
                print('TESTING THE MODEL ====================')
                A.test(model, device, test_loader)
                break

        if data_flag == 'pathmnist':
            # Resnet50 viewing
            print('Resnet50 MODEL LOADING ========================')
            model = B.RestNet50()
            model = model.to(device)
            path = './B/Resnet50'
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print('MODEL LOADED =========================')
            print('MODEL VIEWING ========================')
            print(model)
            print('TESTING THE MODEL ====================')
            B.test(model, device, test_loader)
            break





        

