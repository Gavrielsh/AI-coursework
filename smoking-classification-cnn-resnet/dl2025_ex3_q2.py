import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from clearml import Task

# Initialize ClearML Task
# This allows tracking and logging of training experiments
task = Task.init(project_name='part B project', task_name='fine_tuning_experiment')

# Dataset paths
paths = {
    "regular": {
        "train": "Training",
        "val": "Validation",
        "test": "Testing"
    },
    "augmented": {
        "train": "Training_2",
        "val": "Validation_2",
        "test": "Testing_2"
    }
}


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset class for loading images and corresponding labels.
        """
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith((".jpg", ".png"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load an image, apply transformations, and extract label.
        """
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path).lower()
        label = 1 if filename.startswith("smoking_") else 0
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


# Optimized dataset loader
def load_data(dataset_type="regular", batch_size=64, img_size=256, num_workers=4):
    """
    Loads training, validation, and test data with transformations.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(root=paths[dataset_type]["train"], transform=transform)
    val_dataset = CustomDataset(root=paths[dataset_type]["val"], transform=transform)
    test_dataset = CustomDataset(root=paths[dataset_type]["test"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Load pretrained model for fine-tuning
def load_pretrained_model(num_classes=2):
    """
    Load ResNet18 model and replace the final classification layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extraction layers
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return model


# Optimized training function
def train_model(model, train_loader, val_loader, test_loader, num_epochs=5, lr=0.0001, device="cuda"):
    """
    Train and evaluate the model on training, validation, and test sets.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_correct, val_total, val_loss = 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss {train_loss / len(train_loader):.4f}, Train Acc {100 * correct / total:.2f}% | Val Loss {val_loss / len(val_loader):.4f}, Val Acc {100 * val_correct / val_total:.2f}%")

    # Final evaluation on test set
    test_correct, test_total, test_loss = 0, 0, 0.0
    incorrect_images = []
    model.eval()
    with torch.no_grad():
        for images, labels, img_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Collect incorrect predictions
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    incorrect_images.append(img_paths[i])

    print(f"Final Test Accuracy: {100 * test_correct / test_total:.2f}%, Test Loss: {test_loss / len(test_loader):.4f}")

    # Display up to 5 incorrectly classified images in a single figure
    if incorrect_images:
        fig, axes = plt.subplots(1, min(5, len(incorrect_images)), figsize=(15, 5))
        for i, img_path in enumerate(incorrect_images[:5]):
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].set_title(f"Misclassified: {os.path.basename(img_path)}")
            axes[i].axis("off")
        plt.show()

    return model


# Main execution
if __name__ == "__main__":
    for dataset in ["regular", "augmented"]:
        print(f"\n--- Starting Fine-Tuning on {dataset.capitalize()} Dataset ---\n")
        train_loader, val_loader, test_loader = load_data(dataset, batch_size=64, img_size=256, num_workers=4)
        model = load_pretrained_model()
        trained_model = train_model(model, train_loader, val_loader, test_loader)
        print(f"\n--- Fine-Tuning Completed for {dataset.capitalize()} Dataset ---\n")
