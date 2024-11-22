import torch
import torch.nn as nn
from vit.mhat import MultiHeadAttention

class MLPBlock(nn.Module):
    def __init__(self, hidden_layers: list, drop_out: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Activation {activation} is not supported")

        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            self.act,
            nn.Dropout(drop_out),
            nn.Linear(hidden_layers[1], hidden_layers[0]),
            nn.Dropout(drop_out)
        )

    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        return self.mlp(x, *args, **kwargs)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, hidden_layers: list, num_heads: int, drop_out: float = 0.1, norm_eps: float = 1e-12, activation: str = "gelu"):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm_attn = nn.LayerNorm(normalized_shape=d_model, eps=norm_eps)
        self.mlp = MLPBlock(hidden_layers, drop_out, activation)
        self.norm_mlp = nn.LayerNorm(normalized_shape=d_model, eps=norm_eps)
    
    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        attn_output = self.attn(x, x, x)
        x = x + attn_output
        x = self.norm_attn(x)

        ff_output = self.mlp(x)
        x = x + ff_output
        x = self.norm_mlp(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, mlp_dim: int, num_heads: int, num_layers: int, drop_out: float = 0.1, norm_eps: float = 1e-12, activation: str = "gelu"):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerBlock(d_model=d_model,
                                                      hidden_layers=[d_model, mlp_dim], 
                                                      num_heads=num_heads, 
                                                      drop_out=drop_out, 
                                                      norm_eps=norm_eps, 
                                                      activation=activation)
                                                for _ in range(num_layers)])
    
    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        for i, layer in enumerate(self.encoder):
            x = layer(x, *args, **kwargs)
        return x