# AMLS_assignment23_24

## Deep Learning Models for Medical Image Classification

This repository contains deep learning modeles for training and evaluating on two different medical image datasets: **PneumoniaMNIST** and **PathMNIST**. The tasks are divided into two separate modules, TaskA and TaskB.

In Task A, the code includes implementations of CNN3, LeNet and ResNet18 architectures. The models are trained and evaluated on **PneumoniaMNIST** for a binary classification of Pneumonia and normal X-ray image of lung.

In Task B, the code include implementation of ResNet50 architectures on PathMNIST for a nine-class classification of tissue slides.

## Prerequisites for The project

- Python 3.11
- PyTorch
- torchvision
- Pillow
- pandas
- tqdm
- scikit-learn
- seaborn
- matplotlib
- medmnist (if using online datasets)

## Install the requirements

This should be typed in the **command terminal**.

```
pip install torch torchvision pandas tqdm Pillow seaborn matplotlib medmnist
```

## Dataset Format

**Dataset import are from the save function from medmnist**

For example, download the train dataset of PneumoniaMNIST.

* *from medmnist import PathMNIST, PneumoniaMNIST*
* *train_datasetA = PneumoniaMNIST(split="train",download=True)*
* *train_datasetA.save(folder=".../Datasets/pneumoniamnist/train")*

The format of the dataset would be split into 3 folder which are **train, validate** and **test** respectively. Inside the folder, there are a folder called pneumoniamnist (in this example) which includes all the images in the train dataset and a csv file which includes the name of images and the corresponding label. The main file would extract the label and image from the folder according to the csv file. In order to make this clear, there are the rest code for downloading the pneumoniamnist dataset based on the Dataset format.

* *test_datasetA = PneumoniaMNIST(split="test",download=True)*
* *test_datasetA.save(folder=".../Datasets/pneumoniamnist/test")*
* *val_datasetA = PneumoniaMNIST(split="val",download=True)*
* *val_datasetA.save(folder=".../Datasets/pneumoniamnist/validate")*

The PathMNIST can be download in the same precedure.

* *train_datasetA = PathMNIST(split="train",download=True)*
* *train_datasetA.save(folder=".../Datasets/pathmnistmnist/train")*
* *test_datasetA = PathMNIST(split="test",download=True)*
* *test_datasetA.save(folder=".../Datasets/pathmnistmnist/test")*
* *val_datasetA = PathMNIST(split="val",download=True)*
* *val_datasetA.save(folder=".../Datasets/pathmnist/validate")*

## Role of the file

There are three main file of this project.

* TaskA.py

TaskA.py includes LeNet, CNN3, ResNet18 classes, a train module, a curve module for visualize and a test module.

* TaskB.py

TaskB.py includes ResNet50 class, a train module, a curve module for visualize and a test module.

* main.py

This file is able to call all the class and functions defined in TaskA.py and TaskB.py. There are some instructions and guide when you run the main.py. For example it can let you to choose the dataset import, the module you want to directly check the test performance throught a trained weight applied on the model. The pth files are stored in each task's folder. The results of the model may vary each time based on the device so it may not able to show the best performance in the report.

Inside the file, you may asked to choose the dataset first, you can type 1 or 2 represent for PneumoniaMNIST and PathMNIST. Then you will be asked to choose which action you want to preceed, for viewing the results or for training. You will be able to select the model in PneumoniaMNIST.
