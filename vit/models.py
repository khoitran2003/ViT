import torch
import torch.nn as nn
from vit.embedding import PatchEmbedding
from vit.encoder import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, 
                 num_layers: int = 12,
                 num_heads: int = 12,
                 d_model: int = 768,
                 mlp_dim: int = 3072,
                 num_classes: int = 10,
                 patch_size: int = 16,
                 image_size: int = 224,
                 drop_out: float = 0.1,
                 norm_eps: float = 1e-12,
                 activation: str = "gelu"):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.drop_out = drop_out
        self.norm_eps = norm_eps
        self.activation = activation

        self.embedding = PatchEmbedding(patch_size, image_size, d_model)
        self.encoder = TransformerEncoder(d_model=d_model,
                                          mlp_dim=mlp_dim, 
                                          num_heads=num_heads, 
                                          num_layers=num_layers, 
                                          drop_out=drop_out, 
                                          norm_eps=norm_eps, 
                                          activation=activation)
        self.last_layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=norm_eps)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=d_model, eps=norm_eps),
            nn.Linear(d_model, mlp_dim),
            nn.Dropout(drop_out),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x: torch.tensor):

        # (B, C, H, W) -> (B, N, d_model)
        x = self.embedding(x)

        # (B, N, d_model) -> (B, N, d_model)
        x = self.encoder(x)

        # get cls token
        cls_token = x[:, 0, :] # (B, d_model)

        # (B, d_model) -> (B, num_classes)
        logits = self.mlp_head(cls_token) # (B, num_classes)
        return logits

class ViTBase(ViT):
    def __init__(self,
                 num_classes: int = 10, 
                 patch_size: int = 16, 
                 image_size: int = 224, 
                 drop_out: float = 0.1, 
                 norm_eps: float = 1e-12, 
                 activation = "gelu"):
        super().__init__(num_layers=12, 
                         num_heads=12, 
                         d_model=768, 
                         mlp_dim=3072, 
                         num_classes=num_classes, 
                         patch_size=patch_size, 
                         image_size=image_size, 
                         drop_out=drop_out, 
                         norm_eps=norm_eps, 
                         activation=activation)
    
class ViTLarge(ViT):
    def __init__(self,
                 num_classes: int = 10, 
                 patch_size: int = 16, 
                 image_size: int = 224, 
                 drop_out: float = 0.1, 
                 norm_eps: float = 1e-12, 
                 activation = "gelu"):
        super().__init__(num_layers=24, 
                         num_heads=16, 
                         d_model=1024, 
                         mlp_dim=4096, 
                         num_classes=num_classes, 
                         patch_size=patch_size, 
                         image_size=image_size, 
                         drop_out=drop_out, 
                         norm_eps=norm_eps, 
                         activation=activation)

class ViTHuge(ViT):
    def __init__(self,
                 num_classes: int = 10, 
                 patch_size: int = 16, 
                 image_size: int = 224, 
                 drop_out: float = 0.1, 
                 norm_eps: float = 1e-12, 
                 activation = "gelu"):
        super().__init__(num_layers=32, 
                         num_heads=16, 
                         d_model=1280, 
                         mlp_dim=5120, 
                         num_classes=num_classes, 
                         patch_size=patch_size, 
                         image_size=image_size, 
                         drop_out=drop_out, 
                         norm_eps=norm_eps, 
                         activation=activation)