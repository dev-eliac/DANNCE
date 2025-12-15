import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        # Load pre-trained ViT-Base-16
        self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Replace the head with our specific num_classes
        self.base_model.heads.head = nn.Linear(self.base_model.heads.head.in_features, num_classes)
        
        # ViT-Base hidden dim is 768
        self.output_dim = 768

    def conv_features(self, x):
        # x shape: [B, 3, 224, 224]
        x = self.base_model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.base_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.base_model.encoder(x)

        # Return CLS token: [B, 768]
        x = x[:, 0]
        return x

    def dense_features(self, x):
        return x

    def classifier(self, x):
        return self.base_model.heads.head(x)

    def forward(self, x):
        x = self.conv_features(x)
        x = self.classifier(x)
        return x