from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#Architecture of CNN3
class CNN3(nn.Module):
    def __init__(self, num_classes=1):
        # inherit nn.Module and create CNN3 subclass
        super(CNN3, self).__init__()
        # First Convolution layer with batch normalization and average pool layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # For Second Convolution layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # For Third Convolution Layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Average Pooling layer 
        self.global_avg_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Full connecting layer
        self.fc = nn.Sequential(
            nn.Linear(64 * 14 * 14, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes),
            # Sigmoid function for generate probability prediction
            nn.Sigmoid()
        )

    def forward(self, x):
        # First Layer
        # Convolution layer -> Batch normalization -> ReLU -> Average Pooling Layer
        x = F.relu(self.bn1(self.conv1(x)))
        # Second Layer 
        x = F.relu(self.bn2(self.conv2(x)))
        # Third Layer
        x = F.relu(self.bn3(self.conv3(x)))
        # Global Average Pooling Layer
        x = self.global_avg_pool(x)
        # Flatten accommodate modified pooling layer outputs
        x = x.view(x.size(0), -1)
        # Fully connected Layer
        x = self.fc(x)
        return x

# Architecture of LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
'''
# Architecture of Resnet BasicBlock
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        # inherit nn.Module and create ResnetBasicBlock subclass
        super(ResNetBasicBlock, self).__init__()
        # Construct the structure of Resnet18 BasicBlock
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # Forward propagation function in resnetbasic block
    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

# Architecture of Resnet DownBlock
class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    # Forward propagation function in resnetbasic block
    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

# Architecture of Resnet18
class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64, 1),
                                    ResNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, [2, 1]),
                                    ResNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, [2, 1]),
                                    ResNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, [2, 1]),
                                    ResNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
'''
# Early stopping module
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss is None:
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# Architecture of Resnet BasicBlock
class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # Forward propagation function in resnetbasic block
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# Architecture of Resnet DownBlock
class ResNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.extra(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResNetBasicBlock(64, 64),
                                    ResNetBasicBlock(64, 64))

        self.layer2 = nn.Sequential(ResNetDownBlock(64, 128, stride=2),
                                    ResNetBasicBlock(128, 128))

        self.layer3 = nn.Sequential(ResNetDownBlock(128, 256, stride=2),
                                    ResNetBasicBlock(256, 256))

        self.layer4 = nn.Sequential(ResNetDownBlock(256, 512, stride=2),
                                    ResNetBasicBlock(512, 512))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

# Train module Function
def train_model(model, device, train_loader, val_loader, optimizer, criterion, scheduler, NUM_EPOCHS):

    # Initialize variables to track training and validation metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Iteration start
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        train_loss = 0

        # Set the model in training mode, enabling functions like Dropout.
        model.train()
        # tqdm for show the progress bar
        for inputs, targets in tqdm(train_loader):
            # loading data and target labels onto the specified computing device.
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Call model to generate results
            outputs = model(inputs)
            # Convert model output to binary prediction using a threshold
            predictions = outputs > 0.5
            targets = targets.unsqueeze(1)
            # BCELoss, CrossEntropyLoss, BCEWithLogisticLoss...
            loss = criterion(outputs, targets.float())

            # Backproporgate to compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()
            # Clean gradient for the next batch
            optimizer.zero_grad()

            # number of training times in total
            train_total += targets.size(0)
            # number of correct prediction
            train_correct += (predictions == targets).sum().item()
            # total loss
            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)
        train_accuracy = (train_correct / train_total) * 100.0

        # Print training accuracy and loss at the end of each epoch
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] (Train) - Loss: {average_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

        # Evaluate the model on the validation dataset
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_loss_one = 0

        # Disable gradient calculation during validation
        # Same as training set with out gradient computation
        with torch.no_grad():
            for val_inputs, val_targets in tqdm(val_loader):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)
                val_predictions = val_outputs > 0.5
                val_targets = val_targets.unsqueeze(1)
                val_loss = criterion(val_outputs, val_targets.float())


                val_total += val_targets.size(0)
                val_correct += (val_predictions == val_targets).sum().item()
                val_loss_one += val_loss.item()

        average_val_loss = val_loss_one / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100.0
        
        # Change learning rate based on scheduler
        scheduler.step()

        # Print validation accuracy and loss at the end of each epoch
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] (Validation) - Loss: {average_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Learning Rate: {optimizer.param_groups[0]["lr"]}')
        # Record training and validation losses and accuracies
        train_losses.append(average_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)

        if early_stopping(average_val_loss):
            print('Early stopping triggered. Training Stoped.')
            break

    return train_losses, train_accuracies, val_losses, val_accuracies, epoch


# Visulization module
def curve(train_losses, train_accuracies, val_losses, val_accuracies, epoch_number):

    # Create an epoch list
    epochs = list(range(1,  epoch_number+2))

    # Plot the training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Test Module
def test(model, device, data_loader):
    model.eval()

    running_corrects = 0

    all_preds = []
    all_labels = []

    # Same as validation set without loss computation
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            #outputs = outputs.sequeeze()
            targets = targets.unsqueeze(1)
            targets = targets.float()

            preds = outputs > 0.5

            running_corrects += torch.sum(preds == targets.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

        
        # Calculate the benchmark of the model in test dataset
        epoch_acc = running_corrects.float() / len(data_loader.dataset)
        epoch_auc = roc_auc_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds)
        epoch_recall = recall_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        
        # print the benchmark of the model
        print('Acc: {:.4f} Auc: {:4f} precision: {:4f} recall: {:4f} f1: {:4f}'.format(epoch_acc, epoch_auc, epoch_precision, epoch_recall, epoch_f1))

        # print the confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Print cofusion matrix in the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

