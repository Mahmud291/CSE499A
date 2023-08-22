import os
import timeit
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from main import get_color_distortion, RandomGaussianBlur
from sklearn.model_selection import train_test_split
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using %s device.' % (device))

ROOT_DIR = 'dataset'

# Hyperparameters
BATCH_SIZE  = 32
LEARNING_RATE = 0.001
NUM_CLASSES = 4
num_epochs = 12

# build the augmentations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Compose([
        get_color_distortion(),
        RandomGaussianBlur(),
        ]),
    transforms.ToTensor(),
    normalize,
    ])

# Define dataset and dataloaders
dataset = datasets.ODIR5K(ROOT_DIR, transform)
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load ResNet-150
resnet150 = torch.hub.load('pytorch/vision:v0.11.1', 'resnet152', pretrained=True)

num_features = resnet150.fc.in_features
resnet150.fc = nn.Sequential(
        nn.Linear(num_features, NUM_CLASSES),
        nn.Sigmoid())

resnet150 = resnet150.to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet150.parameters(), lr=LEARNING_RATE)

# Train the model
def train():
    resnet150.train()
    for epoch in range(num_epochs):
        start_time = timeit.default_timer()
        y_true = torch.FloatTensor()
        y_pred = torch.FloatTensor()
        train_loss = 0
        for index, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = resnet150(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true = torch.cat((y_true, labels.cpu()))
            y_pred = torch.cat((y_pred, outputs.detach().cpu()))

            print('\repoch %3d/%3d batch %3d/%3d' % (epoch+1, num_epochs, index, len(train_loader)), end='')
            print(' --- loss %6.4f' % (train_loss / index), end='')
            print(' --- %5.1fsec' % (timeit.default_timer() - start_time), end='')

        aucs = [metrics.roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(NUM_CLASSES)]
        auc_classes = ' '.join(['%5.3f' % (aucs[i]) for i in range(NUM_CLASSES)])
        print(' --- mean AUC score: %5.3f (%s)' % (np.mean(aucs), auc_classes))

train()

# Load pre-trained ResNet-50 as the teacher
teacher = models.resnet50(pretrained=True)

# Clone ResNet-150 to create the student
student = models.resnet150()
student.load_state_dict(resnet150.state_dict())  # Copy weights

# Define distillation loss (e.g., Mean Squared Error)
distillation_criterion = nn.MSELoss()

def distillation_loss(y, teacher_scores, temp):
    soft_teacher = nn.functional.softmax(teacher_scores / temp, dim=1)
    soft_student = nn.functional.log_softmax(y / temp, dim=1)
    return distillation_criterion(soft_student, soft_teacher)

# Knowledge Distillation training
def train_distillation():
    student.train()
    teacher.eval()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs_student = student(images)
            outputs_teacher = teacher(images)
            loss_distillation = distillation_loss(outputs_student, outputs_teacher, temperature)
            loss_distillation.backward()
            optimizer.step()

train_distillation()

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

student_accuracy = evaluate(student, val_loader)
print(f"Student Model Accuracy: {student_accuracy}")
