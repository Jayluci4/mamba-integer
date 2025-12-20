
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dyadic_winograd import DyadicWinograd2D
import time
import copy

# Force use of our CUDA extension if available (it is loaded by dyadic_winograd)

def replace_conv2d_with_dyadic(module):
    replaces = []
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (3, 3) and child.groups == 1 and child.stride == (1, 1):
                new_layer = DyadicWinograd2D(
                    child.in_channels,
                    child.out_channels,
                    m=8,
                    r=3,
                    stride=child.stride[0],
                    padding=child.padding[0]
                )
                new_layer.set_spatial_weights(child.weight)
                replaces.append((name, new_layer))
        else:
            replace_conv2d_with_dyadic(child)
    for name, new_layer in replaces:
        setattr(module, name, new_layer)

def main():
    print("Loading ResNet18...")
    model = timm.create_model('resnet18', pretrained=True)
    
    # Freeze backbone initially? No, let's train the Dyadic layers.
    # But first replace.
    print("Converting to Dyadic-Cayley Model...")
    replace_conv2d_with_dyadic(model)
    model = model.cuda()
    
    # Dataset
    print("Loading ImageNetV2 for Fine-tuning...")
    dataset = load_dataset("vaishaal/image_net_v2", split="train")
    
    # Split into Train (80%) and Val (20%)
    # ImageNetV2 has 10,000 images.
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Simple split indices
    indices = list(range(total_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224), # Augmentation for training
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def collate_fn(batch, tfm):
        images = []
        labels = []
        for item in batch:
            img = tfm(item['jpeg'].convert("RGB"))
            images.append(img)
            class_id = int(item['__key__'].split('/')[-2])
            labels.append(class_id)
        return torch.stack(images), torch.tensor(labels)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, 
                              collate_fn=lambda b: collate_fn(batch=b, tfm=transform))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, 
                            collate_fn=lambda b: collate_fn(batch=b, tfm=val_transform))

    # Optimizer
    # Only train the Dyadic weights (complex frequency weights)
    # This proves we can adapt the spectral domain.
    params_to_update = []
    for name, param in model.named_parameters():
        if 'weight_re' in name or 'weight_im' in name:
            param.requires_grad = True
            params_to_update.append(param)
        else:
            param.requires_grad = False # Freeze standard layers (bn, linear, etc)
            
    optimizer = optim.AdamW(params_to_update, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Fine-tuning {len(params_to_update)} Dyadic Layers (Freezing Backbone)...")
    
    # Evaluate Before
    def evaluate(loader, desc="Val"):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.cuda(), lbls.cuda()
                outputs = model(imgs)
                _, preds = outputs.max(1)
                correct += preds.eq(lbls).sum().item()
                total += lbls.size(0)
                if total > 200: break # Quick check
        acc = correct / total
        print(f"{desc} Accuracy: {acc:.2%}")
        return acc

    print("Initial Accuracy:")
    evaluate(val_loader, "Pre-tune Val")
    
    # Train 1 Epoch (or less for speed)
    print("Starting Training...")
    model.train()
    limit_batches = 50
    for i, (imgs, lbls) in enumerate(train_loader):
        if i >= limit_batches: break
        
        imgs, lbls = imgs.cuda(), lbls.cuda()
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}/{limit_batches}, Loss: {loss.item():.4f}")
            
    print("Training Complete.")
    evaluate(val_loader, "Post-tune Val")

if __name__ == "__main__":
    main()
