"""
FIXED Deep Learning Models with PROPER Video-Level Splitting
==============================================================
Implements DenseNet121, EfficientNetB3, and Xception with:
1. Video-level splitting (no data leakage)
2. Limited frames per video (reduces redundancy)
3. Proper train/val/test splits
4. FEATURE EXTRACTION MODE:
   - Pretrained backbones FROZEN
   - Only train new classifier head
   - Prevents overfitting on small datasets
5. REGULARIZATION to prevent overfitting:
   - Dropout layers (0.5 rate)
   - Weight decay (L2 regularization)
   - Batch normalization
   - Additional hidden layer with ReLU activation
6. DATA AUGMENTATION to improve generalization:
   - Random crops
   - Random flips
   - Color jitter
   - Random rotation
7. Early stopping to prevent overfitting
8. Shuffled training data to prevent ordering bias
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score, 
                             precision_score, recall_score, f1_score)
from collections import defaultdict
import re
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available - EfficientNet and Xception will be skipped")
    print("   Install with: pip install timm")

# Configuration
REAL_PATH = "C://Users//Sinchan A//Desktop//Internship//vid//real"
FAKE_PATH = "C://Users//Sinchan A//Desktop//Internship//vid//fake"

# ============= ADJUST THIS TO CHANGE DATASET SIZE =============
# Current available: ~4600 real + ~4300 fake frames from 70 videos each
# MAX_FRAMES_PER_VIDEO = 3  -->  ~420 total samples  (Low correlation, high generalization)
# MAX_FRAMES_PER_VIDEO = 5  -->  ~700 total samples  (Moderate correlation) ← OPTIMAL BALANCE
# MAX_FRAMES_PER_VIDEO = 8  --> ~1120 total samples  (Higher correlation)
# MAX_FRAMES_PER_VIDEO = 10 --> ~1400 total samples  (Very high correlation)
MAX_FRAMES_PER_VIDEO = 5  # ← Change this number to adjust dataset size
# ==============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 20  # Increased from 10, with early stopping to prevent overfitting
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for 5 epochs
LEARNING_RATE = 0.01  # Higher LR since we're only training classifier (not full network)
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("DEEP LEARNING DEEPFAKE DETECTION - VIDEO-LEVEL SPLIT")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Device: {DEVICE}")
print(f"  Max frames per video: {MAX_FRAMES_PER_VIDEO}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {NUM_EPOCHS}")
print(f"  Early stopping patience: {EARLY_STOP_PATIENCE}")
print(f"  Learning rate: {LEARNING_RATE}")
print("=" * 80)

# Set random seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def extract_video_id(filename):
    """Extract video ID from filename for grouping"""
    name = filename.replace('.jpg', '').replace('.png', '')
    
    # Fake pattern: frame_id0_id1_0001_0 -> id0_id1_0001
    fake_match = re.search(r'frame_(.+?)_\d+$', name)
    if fake_match and 'id' in fake_match.group(1):
        return fake_match.group(1)
    
    # Real pattern: frame_00001_0 -> 00001
    real_match = re.search(r'frame_(\d+)_\d+$', name)
    if real_match:
        return real_match.group(1)
    
    return name.rsplit('_', 1)[0] if '_' in name else name


def group_frames_by_video(image_paths):
    """Group image paths by their source video"""
    video_groups = defaultdict(list)
    for path in image_paths:
        filename = os.path.basename(path)
        video_id = extract_video_id(filename)
        video_groups[video_id].append(path)
    return video_groups


def sample_frames_from_videos(video_groups, max_frames=MAX_FRAMES_PER_VIDEO):
    """Sample limited frames from each video to reduce redundancy"""
    sampled_paths = []
    for video_id, frames in video_groups.items():
        if len(frames) <= max_frames:
            sampled_paths.extend(frames)
        else:
            indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            sampled_paths.extend([frames[i] for i in indices])
    return sampled_paths


def split_videos(video_groups, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split videos into train/val/test sets"""
    video_ids = list(video_groups.keys())
    np.random.shuffle(video_ids)
    
    n_videos = len(video_ids)
    n_train = int(n_videos * train_ratio)
    n_val = int(n_videos * val_ratio)
    
    train_vids = video_ids[:n_train]
    val_vids = video_ids[n_train:n_train+n_val]
    test_vids = video_ids[n_train+n_val:]
    
    return train_vids, val_vids, test_vids


def get_frames_from_video_ids(video_dict, video_ids, max_frames):
    """Get frames for specified video IDs"""
    frames = []
    for vid_id in video_ids:
        vid_frames = video_dict[vid_id]
        if len(vid_frames) <= max_frames:
            frames.extend(vid_frames)
        else:
            indices = np.linspace(0, len(vid_frames)-1, max_frames, dtype=int)
            frames.extend([vid_frames[i] for i in indices])
    return frames


def load_dataset_with_video_split():
    """Load dataset with proper video-level splitting"""
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND SPLITTING DATA")
    print("=" * 80)
    
    # Get all image paths
    real_files = [os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH) 
                  if f.endswith(('.jpg', '.png'))]
    fake_files = [os.path.join(FAKE_PATH, f) for f in os.listdir(FAKE_PATH) 
                  if f.endswith(('.jpg', '.png'))]
    
    print(f"\nTotal frames: {len(real_files)} real + {len(fake_files)} fake")
    
    # Group by video
    real_videos = group_frames_by_video(real_files)
    fake_videos = group_frames_by_video(fake_files)
    
    print(f"Unique videos: {len(real_videos)} real + {len(fake_videos)} fake")
    
    # Split videos
    real_train_vids, real_val_vids, real_test_vids = split_videos(real_videos)
    fake_train_vids, fake_val_vids, fake_test_vids = split_videos(fake_videos)
    
    print(f"\nVideo split:")
    print(f"  Train: {len(real_train_vids)} real + {len(fake_train_vids)} fake videos")
    print(f"  Val:   {len(real_val_vids)} real + {len(fake_val_vids)} fake videos")
    print(f"  Test:  {len(real_test_vids)} real + {len(fake_test_vids)} fake videos")
    
    # Get frames for each split
    train_real = get_frames_from_video_ids(real_videos, real_train_vids, MAX_FRAMES_PER_VIDEO)
    train_fake = get_frames_from_video_ids(fake_videos, fake_train_vids, MAX_FRAMES_PER_VIDEO)
    
    val_real = get_frames_from_video_ids(real_videos, real_val_vids, MAX_FRAMES_PER_VIDEO)
    val_fake = get_frames_from_video_ids(fake_videos, fake_val_vids, MAX_FRAMES_PER_VIDEO)
    
    test_real = get_frames_from_video_ids(real_videos, real_test_vids, MAX_FRAMES_PER_VIDEO)
    test_fake = get_frames_from_video_ids(fake_videos, fake_test_vids, MAX_FRAMES_PER_VIDEO)
    
    # Combine and create labels - SHUFFLE to prevent ordering bias
    X_train = train_real + train_fake
    y_train = [1] * len(train_real) + [0] * len(train_fake)
    
    X_val = val_real + val_fake
    y_val = [1] * len(val_real) + [0] * len(val_fake)
    
    X_test = test_real + test_fake
    y_test = [1] * len(test_real) + [0] * len(test_fake)
    
    # Shuffle train data to prevent class ordering patterns
    import random
    random.seed(RANDOM_STATE)
    train_combined = list(zip(X_train, y_train))
    random.shuffle(train_combined)
    X_train, y_train = zip(*train_combined)
    X_train, y_train = list(X_train), list(y_train)
    
    print(f"\nFrame counts:")
    print(f"  Train: {len(X_train)} frames ({len(train_real)} real + {len(train_fake)} fake) - SHUFFLED")
    print(f"  Val:   {len(X_val)} frames ({len(val_real)} real + {len(val_fake)} fake)")
    print(f"  Test:  {len(X_test)} frames ({len(test_real)} real + {len(test_fake)} fake)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# Custom Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), self.labels[idx]


# Data transforms with augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize larger first
    transforms.RandomCrop((224, 224)),  # Random crop for variation
    transforms.RandomHorizontalFlip(p=0.5),  # Random flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variation
    transforms.RandomRotation(10),  # Slight rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms - no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Model Definitions with Proper Regularization AND Feature Extraction
class DenseNetModel(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.5, freeze_backbone=True):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        
        # FREEZE pretrained layers - only train classifier
        if freeze_backbone:
            for param in self.densenet.features.parameters():
                param.requires_grad = False
            print("  ✓ Backbone frozen - using as feature extractor")
        
        n_features = self.densenet.classifier.in_features
        
        # Replace classifier with regularized version
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.densenet(x)


class EfficientNetModel(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.5, freeze_backbone=True):
        super(EfficientNetModel, self).__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for EfficientNet")
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
        
        # FREEZE pretrained layers - only train classifier
        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            print("  ✓ Backbone frozen - using as feature extractor")
        
        self.n_features = self.efficientnet.classifier.in_features
        
        # Replace classifier with regularized version
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, n_classes)
        )
        
        # Unfreeze the new classifier
        for param in self.efficientnet.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.efficientnet(x)


class XceptionModel(nn.Module):
    def __init__(self, n_classes=2, dropout_rate=0.5, freeze_backbone=True):
        super(XceptionModel, self).__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Xception")
        self.xception = timm.create_model('xception', pretrained=True)
        
        # FREEZE pretrained layers - only train classifier
        if freeze_backbone:
            for param in self.xception.parameters():
                param.requires_grad = False
            print("  ✓ Backbone frozen - using as feature extractor")
        
        n_features = self.xception.fc.in_features
        
        # Replace classifier with regularized version
        self.xception.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, n_classes)
        )
        
        # Unfreeze the new classifier
        for param in self.xception.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.xception(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100*correct/total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Get probabilities for metrics
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs


def train_model(model, model_name, train_loader, val_loader, test_loader, num_epochs=NUM_EPOCHS):
    """Train and evaluate a model"""
    print("\n" + "=" * 80)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 80)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"  Frozen:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    criterion = nn.CrossEntropyLoss()
    # Add weight decay (L2 regularization) to prevent overfitting
    # Only optimize trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=LEARNING_RATE, weight_decay=1e-4)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    print(f"\nTraining for up to {num_epochs} epochs (early stopping patience: {EARLY_STOP_PATIENCE})...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"✓ New best validation accuracy: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                print(f"   Best validation accuracy: {best_val_acc:.2f}% at epoch {epoch + 1 - EARLY_STOP_PATIENCE}")
                break
    
    training_time = time.time() - start_time
    actual_epochs = epoch + 1
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"  Total epochs: {actual_epochs}/{num_epochs}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)
    
    # Final evaluation on all sets
    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION - {model_name.upper()}")
    print("=" * 80)
    
    results = {}
    for split_name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
        loss, acc, preds, labels, probs = validate(model, loader, criterion, DEVICE)
        
        # Calculate metrics
        precision = precision_score(labels, preds, average='binary')
        recall = recall_score(labels, preds, average='binary')
        f1 = f1_score(labels, preds, average='binary')
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        results[split_name] = {
            'loss': loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs,
            'fpr': fpr,
            'tpr': tpr
        }
        
        print(f"\n{split_name} Set:")
        print(f"  Loss:      {loss:.4f}")
        print(f"  Accuracy:  {acc:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return history, results


def visualize_results(model_name, history, results):
    """Create visualizations for a model"""
    print(f"\n" + "=" * 80)
    print(f"GENERATING VISUALIZATIONS - {model_name.upper()}")
    print("=" * 80)
    
    # 1. Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {model_name}_training_history.png")
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, split in enumerate(['Train', 'Validation', 'Test']):
        cm = confusion_matrix(results[split]['labels'], results[split]['predictions'])
        
        im = axes[idx].imshow(cm, cmap='Blues')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Fake', 'Real'])
        axes[idx].set_yticklabels(['Fake', 'Real'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, str(cm[i, j]), ha='center', va='center', 
                             fontsize=14, fontweight='bold')
        
        axes[idx].set_title(f'{split}\nAcc: {results[split]["accuracy"]:.2f}%', 
                          fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.suptitle(f'{model_name} - Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {model_name}_confusion_matrices.png")
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(10, 8))
    
    colors = {'Train': '#2ecc71', 'Validation': '#3498db', 'Test': '#e74c3c'}
    
    for split in ['Train', 'Validation', 'Test']:
        plt.plot(results[split]['fpr'], results[split]['tpr'], 
                label=f'{split} (AUC = {results[split]["roc_auc"]:.4f})',
                color=colors[split], linewidth=2.5)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.savefig(f'{model_name}_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {model_name}_roc_curves.png")
    plt.close()


def main():
    """Main execution pipeline"""
    print(f"\nStarting deep learning models training...")
    print(f"Device: {DEVICE}")
    
    total_start = time.time()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_with_video_split()
    
    # Create datasets with appropriate transforms
    train_dataset = ImageDataset(X_train, y_train, transform=train_transform)  # With augmentation
    val_dataset = ImageDataset(X_val, y_val, transform=val_transform)  # No augmentation
    test_dataset = ImageDataset(X_test, y_test, transform=val_transform)  # No augmentation
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Define models to train
    models_to_train = [
        ('DenseNet121', DenseNetModel),
    ]
    
    if TIMM_AVAILABLE:
        models_to_train.extend([
            ('EfficientNetB3', EfficientNetModel),
            ('Xception', XceptionModel),
        ])
    
    all_results = {}
    
    # Train each model
    for model_name, ModelClass in models_to_train:
        try:
            print(f"\n\n{'='*80}")
            print(f"MODEL: {model_name}")
            print(f"{'='*80}")
            
            model = ModelClass(n_classes=2).to(DEVICE)
            history, results = train_model(model, model_name, train_loader, val_loader, test_loader)
            visualize_results(model_name, history, results)
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            continue
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 80)
    
    print("\nTest Set Performance:")
    print("-" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        test_res = results['Test']
        print(f"{model_name:<20} {test_res['accuracy']:>10.2f}%  {test_res['f1']:>10.4f}  {test_res['roc_auc']:>10.4f}")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
