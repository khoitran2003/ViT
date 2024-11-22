import torch.nn as nn
import torch

class Patches():
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, x):
        """
        B: Batch size
        C: Channel
        H: Height
        W: Width
        P: Patch size

        Pass images in to pathes
        :param x: [B, C H, W,]
        :return: [B, N, C*P^2] 

        N = (HW)//(P^2) number of patches
        H = W (resize image to 244, 224 or 384, 384)
        """
        B, C, H, W = x.shape

        # (B, C, H/P, W/P, P, P)
        img_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)

        # (B, C, H/P, W/P, P, P) -> (B, H/P, W/P, C, P, P)
        img_patches = img_patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        # (B, H/P, W/P, C, P, P) -> (B, N, C*P*P)
        img_patches = img_patches.view(B, -1, C * self.patch_size * self.patch_size)

        return img_patches

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, image_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Linear(patch_size * patch_size * 3, d_model)
        self.patches = Patches(patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # (1, 1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model)) # (1, num_patches + 1, d_model)
    def forward(self, x: torch.tensor):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B, N, C*P*P)
        x = self.patches(x)

        # (B, N, C*P*P) -> (B, N, d_model)
        x = self.projection(x)

        # combine cls token and patches
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # (B, N + 1, d_model)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position encoding
        x = x + self.position_encoding
        return x # (B, N + 1, d_model)