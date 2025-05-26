import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from clearml import Task

# Initialize ClearML Task
task = Task.init(project_name='part A project', task_name='experiment 1')

# Custom Dataset class for loading data
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                if filename.startswith("smoking"):
                    self.images.append(file_path)
                    self.labels.append(0)
                elif filename.startswith("notsmoking"):
                    self.images.append(file_path)
                    self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Function to display dataset examples
def show_examples(dataset, num_examples=5):
    categories = {0: "Smoking", 1: "Not Smoking"}
    indices = random.sample(range(len(dataset)), num_examples)
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].set_title(categories[label])
        axes[i].axis("off")
    plt.show()

# Train one epoch
def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    errors = []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                errors.append((images[i].cpu(), labels[i].cpu(), predicted[i].cpu()))

    loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return loss, accuracy, errors

# Evaluate model
def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return loss, accuracy

# Plot misclassified images
def plot_misclassified(errors, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, (image, true_label, predicted_label) in enumerate(errors[:num_images]):
        image = transforms.ToPILImage()(image)
        axes[i].imshow(image)
        axes[i].set_title(f"True: {true_label.item()}\nPred: {predicted_label.item()}")
        axes[i].axis("off")
    plt.suptitle("Misclassified Images", fontsize=16)
    plt.show()

# Define CNN
def create_cnn(num_classes):
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Update the input size to fc1
            self.fc1 = nn.Linear(128 * 43 * 43, 256)  # Adjusted for 344x344 input
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = F.relu(self.bn1(self.pool(self.conv1(x))))
            x = F.relu(self.bn2(self.pool(self.conv2(x))))
            x = F.relu(self.bn3(self.pool(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)

    return CNN()

# Set paths and parameters
"""
paths = {
    "train": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Training",
    "val": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Validation",
    "test": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Testing"
}
"""

paths_2 = {
    "train": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Training_2",
    "val": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Validation_2",
    "test": "C:/Users/gavri/Desktop/neural network/ex4/student_205461486/Testing_2"
}

transform = transforms.Compose([
    transforms.Resize((344, 344)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define experiments with different parameters
experiments = [
    #{"optimizer": "SGD", "learning_rate": 0.001, "momentum": 0.9, "batch_size": 8, "augmentation": True},
    #{"optimizer": "SGD", "learning_rate": 0.0001, "momentum": 0.9, "batch_size": 32, "augmentation": True},
    #{"optimizer": "SGD", "learning_rate": 0.005, "momentum": 0.9, "batch_size": 64, "augmentation": False},
    #{"optimizer": "Adam", "learning_rate": 0.005, "batch_size": 64, "augmentation": True},
    #{"optimizer": "Adam", "learning_rate": 0.0005, "batch_size": 16, "augmentation": False}
    {"optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.9, "batch_size": 128, "augmentation": False},
    {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 128, "augmentation": True},
    {"optimizer": "SGD", "learning_rate": 0.1, "momentum": 0.9, "batch_size": 256, "augmentation": True},
    {"optimizer": "Adam", "learning_rate": 0.001, "batch_size": 256, "augmentation": False},
]

num_epochs = 5
"""
# Loop through experiments for regular dataset
print("================ RUNNING ON REGULAR DATASET ================")
for exp in experiments:
    print("==================================================")
    print(f"Running experiment with the following parameters:")
    print(f"Optimizer: {exp['optimizer']}")
    print(f"Learning Rate: {exp['learning_rate']}")
    print(f"Batch Size: {exp['batch_size']}")
    print(f"Augmentation: {exp['augmentation']}")
    print("==================================================")

    # Update parameters
    learning_rate = exp['learning_rate']
    batch_size = exp['batch_size']

    if exp['augmentation']:
        transform = transforms.Compose([
            transforms.Resize((344, 344)),
            transforms.RandomRotation(180),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    # Load data
    train_dataset = CustomDataset(paths["train"], transform)
    val_dataset = CustomDataset(paths["val"], transform)
    test_dataset = CustomDataset(paths["test"], transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = create_cnn(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_name = exp['optimizer'].strip().lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=exp.get('momentum', 0.9))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {exp['optimizer']}")
    results = []
    last_epoch_errors = []  # Store errors from the last epoch

    # Train and validate
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_errors = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        # Save results
        result = (f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        results.append(result)

        print("\nTraining Results:")
        for res in results:
            print(res)

        # Save results to a file
        with open("experiment_results.txt", "a") as f:
            f.write(result + "\n")

        # Store errors from the last epoch only
        if epoch == num_epochs - 1:
            last_epoch_errors = train_errors

    # Plot and save misclassified examples from the last epoch
    if len(last_epoch_errors) > 0:
        print(f"\nMisclassified examples from the last epoch:")
        plot_misclassified(last_epoch_errors, num_images=5)

    # Final test evaluation
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
"""
# Loop through experiments for augmented dataset
print("\n================ RUNNING ON AUGMENTED DATASET ================")

for exp in experiments:
    print("==================================================")
    print(f"Running experiment with the following parameters:")
    print(f"Optimizer: {exp['optimizer']}")
    print(f"Learning Rate: {exp['learning_rate']}")
    print(f"Batch Size: {exp['batch_size']}")
    print(f"Augmentation: {exp['augmentation']}")
    print("==================================================")

    # Validate experiment configuration
    required_keys = {'optimizer', 'learning_rate', 'batch_size', 'augmentation'}
    missing_keys = required_keys - exp.keys()
    if missing_keys:
        raise KeyError(f"Missing keys in experiment configuration: {missing_keys}")

    if exp['optimizer'] not in ["SGD", "Adam"]:
        raise ValueError(f"Unsupported optimizer: {exp['optimizer']}")

    # Update parameters
    learning_rate = exp['learning_rate']
    batch_size = exp['batch_size']

    # Define data transformations
    if exp['augmentation']:
        transform = transforms.Compose([
            transforms.Resize((344, 344)),
            transforms.RandomRotation(180),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((344, 344)),
            transforms.ToTensor()
        ])

    # Load data from the augmented dataset
    train_dataset = CustomDataset(paths_2["train"], transform)
    val_dataset = CustomDataset(paths_2["val"], transform)
    test_dataset = CustomDataset(paths_2["test"], transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = create_cnn(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    if exp['optimizer'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=exp.get('momentum', 0.9))
    elif exp['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {exp['optimizer']}")

    results = []
    last_epoch_errors = []  # Store errors from the last epoch

    # Train and validate
    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_errors = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        # Save results
        result = (f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        results.append(result)

        print("\nTraining Results:")
        for res in results:
            print(res)

        # Save results to a file
        with open("experiment_results.txt", "a") as f:
            f.write(result + "\n")

        # Store errors from the last epoch only
        if epoch == num_epochs - 1:
            last_epoch_errors = train_errors

    # Plot and save misclassified examples from the last epoch
    if len(last_epoch_errors) > 0:
        print(f"\nMisclassified examples from the last epoch:")
        plot_misclassified(last_epoch_errors, num_images=5)


    # Final test evaluation
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    print(f"\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    task.close()
