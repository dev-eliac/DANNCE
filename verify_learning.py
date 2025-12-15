import torch
import torch.nn as nn
import torch.optim as optim
from src.models.vit import VisionTransformer
from src.models.heads import TransformerDiscriminator

def verify_learning():
    print("=== Verifying Model Learning Capability ===")
    
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    num_classes = 7
    num_domains = 3
    batch_size = 4
    
    model = VisionTransformer(num_classes=num_classes).to(device)
    discriminator = TransformerDiscriminator(num_classes=num_domains).to(device)
    
    # Optimizer (Copying exact logic from main.py)
    features_lr = 1e-4 
    classifier_lr = 1e-3
    domain_lr = 1e-3
    
    # FIX: Separate params cleanly
    head_params = list(model.base_model.heads.head.parameters())
    head_ids = list(map(id, head_params))
    backbone_params = filter(lambda p: id(p) not in head_ids, model.base_model.parameters())

    lr_groups = [
        {'params': backbone_params, 'lr': features_lr},
        {'params': head_params, 'lr': classifier_lr},
        {'params': discriminator.parameters(), 'lr': domain_lr}
    ]
    
    optimizer = optim.AdamW(lr_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Synthetic Data
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    y = torch.randint(0, num_classes, (batch_size,)).to(device)
    d = torch.randint(0, num_domains, (batch_size,)).to(device)
    
    print(f"Initial Class Labels: {y}")
    
    # 3. Training Loop (10 steps)
    model.train()
    discriminator.train()
    
    initial_loss = None
    
    for i in range(10):
        optimizer.zero_grad()
        
        # Forward
        features = model.conv_features(x)
        logits = model.classifier(features)
        domain_logits = discriminator(features)
        
        c_loss = criterion(logits, y)
        d_loss = criterion(domain_logits, d)
        
        total_loss = c_loss + d_loss
        
        if i == 0:
            initial_loss = total_loss.item()
            print(f"Step 0 Loss: {total_loss.item():.4f} (Class: {c_loss.item():.4f}, Domain: {d_loss.item():.4f})")
            
        total_loss.backward()
        
        # Gradient Check
        if i == 0:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
            print(f"Gradient Norm (Step 0): {grad_norm:.4f}")
            if grad_norm == 0:
                print("FATAL: No gradients flowing!")
                exit(1)
        
        optimizer.step()
        
    final_loss = total_loss.item()
    print(f"Step 10 Loss: {final_loss:.4f}")
    
    # 4. Verification
    if final_loss < initial_loss:
        print("✅ SUCCESS: Loss decreased.")
        print(f"Improvement: {initial_loss - final_loss:.4f}")
    else:
        print("❌ FAILURE: Loss did not decrease.")
        exit(1)

if __name__ == "__main__":
    verify_learning()
