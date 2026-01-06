"""
Skin Cancer Detection Model Training - PyTorch Version
7-Class Classification using HAM10000 Dataset
dx types: mel (melanoma), nv (nevus), bcc, bkl, akiec, vasc, df
Using Transfer Learning with EfficientNetB4
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configure GPU
print("Configuring GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[OK] CUDA version: {torch.version.cuda}")
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] Number of GPUs: {torch.cuda.device_count()}")
    print(f"[OK] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("[OK] Mixed precision training enabled (FP16)")
else:
    print("[WARNING] No GPU detected - using CPU (training will be slower)")

# Configuration
CONFIG = {
    'metadata_file': 'HAM10000_metadata.csv',
    'image_dir': r'Skin Cancer\Skin Cancer',
    'processed_dir': 'skin_cancer_processed',
    'img_size': (224, 224),
    'batch_size': 64,  # Increased for GPU (reduce to 32 if OOM)
    'epochs': 50,
    'learning_rate': 0.0001,
    'model_name': 'skin_cancer_efficientnet_pytorch',
    'num_classes': 7,
    'dx_mapping': {
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevus',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'akiec': 'Actinic Keratosis',
        'vasc': 'Vascular Lesion',
        'df': 'Dermatofibroma'
    },
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}


class SkinCancerModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        self.metadata = None
        self.class_indices = None
        self.device = device
        
    def prepare_dataset(self):
        """Load metadata and organize images into class folders"""
        print("Loading metadata...")
        self.metadata = pd.read_csv(self.config['metadata_file'])
        
        print(f"Total images in metadata: {len(self.metadata)}")
        print(f"\nClass distribution:")
        print(self.metadata['dx'].value_counts())
        
        # Create processed directory structure
        print(f"\nCreating directory structure in {self.config['processed_dir']}...")
        
        for split in ['train', 'val', 'test']:
            for dx in self.metadata['dx'].unique():
                dx_name = self.config['dx_mapping'].get(dx, dx)
                Path(self.config['processed_dir'], split, dx_name).mkdir(parents=True, exist_ok=True)
        
        # Split data by lesion_id to avoid data leakage (same lesion in train/test)
        unique_lesions = self.metadata.groupby('lesion_id').first().reset_index()
        
        train_lesions, temp_lesions = train_test_split(
            unique_lesions['lesion_id'],
            test_size=(self.config['val_split'] + self.config['test_split']),
            random_state=42,
            stratify=unique_lesions['dx']
        )
        
        val_lesions, test_lesions = train_test_split(
            temp_lesions,
            test_size=self.config['test_split'] / (self.config['val_split'] + self.config['test_split']),
            random_state=42,
            stratify=unique_lesions[unique_lesions['lesion_id'].isin(temp_lesions)]['dx']
        )
        
        # Copy images to respective folders
        print("\nCopying images to train/val/test folders...")
        
        split_counts = {'train': 0, 'val': 0, 'test': 0}
        missing_images = []
        
        for idx, row in self.metadata.iterrows():
            image_id = row['image_id']
            dx = row['dx']
            dx_name = self.config['dx_mapping'].get(dx, dx)
            lesion_id = row['lesion_id']
            
            # Determine split
            if lesion_id in train_lesions.values:
                split = 'train'
            elif lesion_id in val_lesions.values:
                split = 'val'
            else:
                split = 'test'
            
            # Find source image
            source_path = Path(self.config['image_dir'], f"{image_id}.jpg")
            
            if not source_path.exists():
                # Try without extension or different format
                alt_path = Path(self.config['image_dir'], image_id)
                if alt_path.exists():
                    source_path = alt_path
                else:
                    missing_images.append(image_id)
                    continue
            
            # Copy to destination
            dest_path = Path(self.config['processed_dir'], split, dx_name, f"{image_id}.jpg")
            
            if not dest_path.exists():
                shutil.copy2(source_path, dest_path)
                split_counts[split] += 1
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.metadata)} images...")
        
        print(f"\nDataset preparation complete!")
        print(f"Train: {split_counts['train']} images")
        print(f"Validation: {split_counts['val']} images")
        print(f"Test: {split_counts['test']} images")
        
        if missing_images:
            print(f"\nWarning: {len(missing_images)} images not found")
            with open('missing_images.txt', 'w') as f:
                f.write('\n'.join(missing_images))
    
    def create_data_loaders(self):
        """Create train, validation, and test data loaders with augmentation"""
        print("\nCreating data loaders...")
        
        # Training data augmentation
        train_transform = transforms.Compose([
            transforms.Resize(self.config['img_size']),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation/Test data (no augmentation)
        val_test_transform = transforms.Compose([
            transforms.Resize(self.config['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(
            os.path.join(self.config['processed_dir'], 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.config['processed_dir'], 'val'),
            transform=val_test_transform
        )
        
        test_dataset = datasets.ImageFolder(
            os.path.join(self.config['processed_dir'], 'test'),
            transform=val_test_transform
        )
        
        self.class_indices = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_indices.items()}
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Class indices: {self.class_indices}")
        
        # Calculate class weights for imbalanced dataset
        class_counts = np.bincount([label for _, label in train_dataset])
        class_weights = 1. / class_counts
        class_weights = class_weights / class_weights.sum()
        
        # Create sample weights for weighted random sampling
        sample_weights = [class_weights[label] for _, label in train_dataset]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"\nClass weights (for imbalanced dataset):")
        for class_name, class_idx in sorted(self.class_indices.items(), key=lambda x: x[1]):
            print(f"  {class_name}: {class_weights[class_idx]:.3f}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
    def build_model(self):
        """Build EfficientNet-B4 model with custom head"""
        print("\nBuilding model...")
        
        # Load pre-trained EfficientNet-B4
        self.model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        
        # Freeze base model initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Replace classifier head
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.config['num_classes'])
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model built with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, epoch, criterion, optimizer, scaler):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{torch.sum(preds == labels.data).double() / inputs.size(0):.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def validate(self, epoch, criterion):
        """Validate on validation set"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
        
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).double() / inputs.size(0):.4f}'
                })
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.double() / len(self.val_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def train(self):
        """Train the model with callbacks"""
        print("\n" + "="*80)
        print("Training model...")
        print("="*80)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        
        # Optimizer
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate']
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Gradient scaler for mixed precision
        scaler = GradScaler()
        
        # Early stopping
        best_val_acc = 0.0
        patience = 12
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch, criterion, optimizer, scaler)
            
            # Validate
            val_loss, val_acc = self.validate(epoch, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, f"{self.config['model_name']}_best.pth")
                print(f"[OK] Model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Fine-tuning phase
        print("\n" + "="*80)
        print("Fine-tuning model...")
        print("="*80)
        
        # Unfreeze last few layers
        for param in list(self.model.features.parameters())[-50:]:
            param.requires_grad = True
        
        # Lower learning rate for fine-tuning
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['learning_rate'] / 10
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Fine-tuning loop (25 additional epochs)
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(25):
            train_loss, train_acc = self.train_epoch(epoch, criterion, optimizer, scaler)
            val_loss, val_acc = self.validate(epoch, criterion)
            scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            print(f"\nFine-tune Epoch {epoch+1}/25")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, f"{self.config['model_name']}_best.pth")
                print(f"[OK] Model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} fine-tuning epochs")
                break
        
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n" + "="*80)
        print("Evaluating model on test set...")
        print("="*80)
        
        # Load best model
        checkpoint = torch.load(f"{self.config['model_name']}_best.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Classification report
        class_names = [name for name, idx in sorted(self.class_indices.items(), key=lambda x: x[1])]
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate test accuracy
        test_acc = np.mean(all_preds == all_labels)
        
        # Multi-class ROC-AUC
        y_true_bin = label_binarize(all_labels, classes=range(self.config['num_classes']))
        roc_auc_per_class = {}
        for i, class_name in enumerate(class_names):
            roc_auc_per_class[class_name] = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"\nROC-AUC per class:")
        for class_name, auc_score in roc_auc_per_class.items():
            print(f"  {class_name}: {auc_score:.4f}")
        
        # Save results
        results = {
            'test_accuracy': float(test_acc),
            'roc_auc_per_class': {k: float(v) for k, v in roc_auc_per_class.items()},
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
        
        with open(f"{self.config['model_name']}_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names)
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Skin Cancer Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.config['model_name']}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Confusion matrix saved as {self.config['model_name']}_confusion_matrix.png")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f"{self.config['model_name']}_training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Training history saved as {self.config['model_name']}_training_history.png")


def main():
    print("="*80)
    print("SKIN CANCER CLASSIFICATION - PyTorch Version")
    print("="*80)
    
    # Initialize model
    skin_model = SkinCancerModel(CONFIG)
    
    # Check if dataset is already prepared
    train_dir = Path(CONFIG['processed_dir']) / 'train'
    if not train_dir.exists() or len(list(train_dir.glob('*/*.jpg'))) == 0:
        print("\nPreparing dataset...")
        skin_model.prepare_dataset()
    else:
        print(f"\n[OK] Dataset already prepared in {CONFIG['processed_dir']}")
        skin_model.metadata = pd.read_csv(CONFIG['metadata_file'])
    
    # Create data loaders
    skin_model.create_data_loaders()
    
    # Build model
    skin_model.build_model()
    
    # Train model
    print(f"\n{'='*80}")
    print(f"Starting training on {device}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']} + 25 fine-tuning")
    print(f"{'='*80}\n")
    
    history = skin_model.train()
    
    # Evaluate model
    results = skin_model.evaluate()
    
    # Plot training history
    skin_model.plot_training_history()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"[OK] Best model saved as: {CONFIG['model_name']}_best.pth")
    print(f"[OK] Results saved as: {CONFIG['model_name']}_results.json")
    print(f"[OK] Test Accuracy: {results['test_accuracy']:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
