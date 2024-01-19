from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from medmnist import Evaluator
import matplotlib.pyplot as plt


#Architecture of Resnet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # inherit nn.Module and create Bottleneck subclass
        super(Bottleneck, self).__init__()
        # First Convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second Convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Third Convolution layer
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # Convolution layer -> Batch normalization -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class RestNet50(nn.Module):
    def __init__(self):
        super(RestNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, 9)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


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


# Train module
def train_model(model, device, train_loader, val_loader, optimizer, criterion, scheduler, NUM_EPOCHS):

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        train_loss = 0


        model.train()
        for inputs, targets in tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += targets.size(0)
            train_correct += (predictions == targets).sum().item()
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

        with torch.no_grad():
            for val_inputs, val_targets in tqdm(val_loader):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_outputs = model(val_inputs)
                _, val_predictions = torch.max(val_outputs, 1)

                val_targets = val_targets.squeeze().long()
                val_loss = criterion(val_outputs, val_targets)


                val_total += val_targets.size(0)
                val_correct += (val_predictions == val_targets).sum().item()
                val_loss_one += val_loss.item()

        average_val_loss = val_loss_one / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100.0

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
    epochs = list(range(1,  epoch_number+1))

    # Plotting training and validation loss trend
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

# The test module for the Resnet50
def test(model, device, data_loader):
    model.eval()

    y_true_list = []
    y_score_list = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true_list.append(targets.cpu().numpy())
            y_score_list.append(outputs.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_score = np.concatenate(y_score_list)

    # Calculate Confusion matrix
    y_pred = np.argmax(y_score, axis=1)
    confusion = confusion_matrix(y_true, y_pred)

    # Use seaburn to plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Print Classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)