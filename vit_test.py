import torch
from src.models.vit import VisionTransformer
from src.models.heads import TransformerDiscriminator

def test_vit_shapes():
    print("Testing VisionTransformer shapes...")
    batch_size = 4
    num_classes = 7 # PACS has 7 classes
    
    model = VisionTransformer(num_classes=num_classes)
    model.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # Test conv_features
    features = model.conv_features(x)
    print(f"Features shape: {features.shape}")
    assert features.shape == (batch_size, 768), f"Expected (4, 768), got {features.shape}"
    
    # Test classifier
    logits = model.classifier(features)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, num_classes), f"Expected (4, 7), got {logits.shape}"
    
    print("Verification passed! Model interface is compatible.")

def test_vit_system():
    print("Testing Full Transformer System (Model + Discriminator)...")
    batch_size = 4
    num_classes = 7
    num_domains = 3 # PACS has 3 domains if we hold one out
    
    # 1. Setup Model
    model = VisionTransformer(num_classes=num_classes)
    model.eval()
    
    # 2. Setup Discriminator
    # Note: Domain adversary wrapper usually handles the gradient reversal,
    # but the Head just processes features.
    discriminator_head = TransformerDiscriminator(num_classes=num_domains)
    discriminator_head.eval()

    # 3. Create dummy input
    x = torch.randn(batch_size, 3, 224, 224)
    
    # 4. Forward Pass
    features = model.conv_features(x)
    print(f"Features: {features.shape}")
    
    class_logits = model.classifier(features)
    print(f"Class Logits: {class_logits.shape}")
    
    domain_logits = discriminator_head(features)
    print(f"Domain Logits: {domain_logits.shape}")
    
    # 5. Assertions
    assert features.shape == (batch_size, 768)
    assert class_logits.shape == (batch_size, num_classes)
    assert domain_logits.shape == (batch_size, num_domains)
    
    print("Verification passed! Transformer + Discriminator are compatible.")

if __name__ == "__main__":
    test_vit_system()
