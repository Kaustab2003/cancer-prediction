"""
Blood Cell Cancer Detection Model Training - PyTorch CUDA Version
4-Class Classification: Benign, early Pre-B, Pre-B, Pro-B
Using Transfer Learning with EfficientNetB3
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GPU Configuration
print("Configuring GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] Number of GPUs: {torch.cuda.device_count()}")
    print(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("[OK] Mixed precision training enabled (FP16)")
else:
    print("[WARNING] No GPU detected - using CPU (training will be slower)")

print("="*80)
print("BLOOD CELL CANCER DETECTION - PyTorch Version")
print("="*80)

# Configuration
CONFIG = {
    'data_dir': r'Blood cell Cancer [ALL]',
    'img_size': 224,
    'batch_size': 64 if torch.cuda.is_available() else 16,
    'epochs': 50,
    'learning_rate': 0.0001,
    'model_name': 'blood_cancer_efficientnet_pytorch',
    'num_classes': 4,
    'class_names': ['Benign', '[Malignant] early Pre-B', '[Malignant] Pre-B', '[Malignant] Pro-B'],
    'validation_split': 0.2,
    'test_split': 0.1,
    'num_workers': 4 if torch.cuda.is_available() else 0,
    'device': device
}


class BloodCancerDataset(Dataset):
    """Custom Dataset for Blood Cell Cancer images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class EfficientNetB3Model(nn.Module):
    """EfficientNetB3 with custom head for blood cancer classification"""
    
    def __init__(self, num_classes=4):
        super(EfficientNetB3Model, self).__init__()
        
        # Load pre-trained EfficientNetB3
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # Get the number of features from the last layer
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone layers"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, num_layers=50):
        """Unfreeze last N layers of backbone"""
        # Unfreeze all first
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        
        # Then freeze all except last num_layers
        total_layers = len(list(self.backbone.features.children()))
        layers_to_freeze = total_layers - num_layers
        
        for i, child in enumerate(self.backbone.features.children()):
            if i < layers_to_freeze:
                for param in child.parameters():
                    param.requires_grad = False


class BloodCancerTrainer:
    """Training pipeline for Blood Cancer Detection"""
    
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_to_idx = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def prepare_data(self):
        """Prepare datasets and dataloaders"""
        print("\nPreparing datasets...")
        
        data_dir = Path(self.config['data_dir'])
        
        # Collect all images and labels
        image_paths = []
        labels = []
        class_names = []
        
        for class_dir in sorted(data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                class_names.append(class_name)
                class_idx = len(class_names) - 1
                
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_paths.append(str(img_path))
                        labels.append(class_idx)
        
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        print(f"Classes found: {class_names}")
        print(f"Total images: {len(image_paths)}")
        
        # Split data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, test_size=self.config['validation_split'] + self.config['test_split'],
            stratify=labels, random_state=42
        )
        
        val_size = self.config['validation_split'] / (self.config['validation_split'] + self.config['test_split'])
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(1 - val_size),
            stratify=temp_labels, random_state=42
        )
        
        print(f"Training samples: {len(train_paths)}")
        print(f"Validation samples: {len(val_paths)}")
        print(f"Test samples: {len(test_paths)}")
        
        # Data augmentation transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = BloodCancerDataset(train_paths, train_labels, train_transform)
        val_dataset = BloodCancerDataset(val_paths, val_labels, val_test_transform)
        test_dataset = BloodCancerDataset(test_paths, test_labels, val_test_transform)
        
        # Calculate class weights for balanced sampling
        class_counts = np.bincount(train_labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print("[OK] Data preparation complete\n")
        
    def build_model(self):
        """Build and initialize model"""
        print("Building model...")
        
        self.model = EfficientNetB3Model(num_classes=self.config['num_classes'])
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("[OK] Model built successfully\n")
        
    def train_epoch(self, optimizer, criterion, scaler):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = self.model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("="*80)
        print("Starting training...")
        print("="*80)
        
        # Freeze backbone initially
        self.model.freeze_backbone()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        scaler = GradScaler()
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        # Phase 1: Train with frozen backbone
        print("\nPhase 1: Training with frozen backbone (30 epochs)")
        print("-" * 80)
        
        for epoch in range(30):
            print(f"\nEpoch {epoch + 1}/30")
            
            train_loss, train_acc = self.train_epoch(optimizer, criterion, scaler)
            val_loss, val_acc = self.validate(criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, f"{self.config['model_name']}_best.pth")
                print(f"[OK] Best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Phase 2: Fine-tuning
        print("\n" + "="*80)
        print("Phase 2: Fine-tuning (unfreezing last 50 layers)")
        print("="*80)
        
        self.model.unfreeze_backbone(num_layers=50)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'] / 10)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        patience_counter = 0
        
        for epoch in range(20):
            print(f"\nFine-tune Epoch {epoch + 1}/20")
            
            train_loss, train_acc = self.train_epoch(optimizer, criterion, scaler)
            val_loss, val_acc = self.validate(criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, f"{self.config['model_name']}_best.pth")
                print(f"[OK] Best model saved! Val Acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} fine-tune epochs")
                break
        
        print("\n[OK] Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*80)
        print("Evaluating on test set...")
        print("="*80)
        
        # Load best model
        checkpoint = torch.load(f"{self.config['model_name']}_best.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = 100. * np.mean(all_preds == all_labels)
        
        # Get class names
        idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC-AUC per class
        y_true_bin = label_binarize(all_labels, classes=range(self.config['num_classes']))
        roc_auc_per_class = {}
        for i, class_name in enumerate(class_names):
            roc_auc_per_class[class_name] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        
        # Overall AUC
        overall_auc = roc_auc_score(y_true_bin, all_probs, average='weighted')
        
        results = {
            'test_accuracy': float(accuracy),
            'test_auc': float(overall_auc),
            'roc_auc_per_class': roc_auc_per_class,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names,
            'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        }
        
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        print(f"Test AUC: {overall_auc:.4f}")
        print("\nROC-AUC per class:")
        for class_name, auc in roc_auc_per_class.items():
            print(f"  {class_name}: {auc:.4f}")
        
        return results, all_labels, all_preds, all_probs
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['model_name']}_training_history.png", dpi=300, bbox_inches='tight')
        print(f"\n[OK] Training history plot saved: {self.config['model_name']}_training_history.png")
        plt.close()
        
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Blood Cell Cancer - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.config['model_name']}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix saved: {self.config['model_name']}_confusion_matrix.png")
        plt.close()
        
    def save_results(self, results):
        """Save all results and metadata"""
        print("\nSaving results...")
        
        # Save metadata
        metadata = {
            'model_name': self.config['model_name'],
            'cancer_type': 'Blood Cell Cancer (ALL)',
            'num_classes': self.config['num_classes'],
            'class_names': results['class_names'],
            'class_to_idx': self.class_to_idx,
            'input_shape': [3, self.config['img_size'], self.config['img_size']],
            'test_accuracy': results['test_accuracy'],
            'test_auc': results['test_auc'],
            'roc_auc_per_class': results['roc_auc_per_class'],
            'training_date': datetime.now().isoformat(),
            'device': str(self.device),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'architecture': 'EfficientNetB3 + Custom Head (PyTorch)',
            'total_epochs': len(self.history['train_loss']),
            'best_val_accuracy': max(self.history['val_acc']),
        }
        
        with open(f"{self.config['model_name']}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[OK] Metadata saved: {self.config['model_name']}_metadata.json")
        
        # Save results
        with open(f"{self.config['model_name']}_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[OK] Results saved: {self.config['model_name']}_results.json")
        
        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(f"{self.config['model_name']}_history.csv", index=False)
        print(f"[OK] Training history saved: {self.config['model_name']}_history.csv")
        
        print("\n[OK] All results saved successfully!")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("Blood Cell Cancer Detection - PyTorch CUDA Training")
    print("4-Class Classification: Benign, early Pre-B, Pre-B, Pro-B")
    print("="*80 + "\n")
    
    # Initialize trainer
    trainer = BloodCancerTrainer(CONFIG)
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    results, y_true, y_pred, y_probs = trainer.evaluate()
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(np.array(results['confusion_matrix']), results['class_names'])
    
    # Save results
    trainer.save_results(results)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print("\nROC-AUC per class:")
    for class_name, auc in results['roc_auc_per_class'].items():
        print(f"  {class_name}: {auc:.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
