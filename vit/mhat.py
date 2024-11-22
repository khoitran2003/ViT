import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Params:
            d_model: dimension of the model
            num_heads: number of heads
            embed_dim: dimension of the embeddings
        """
        super().__init__()
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None) -> tuple[torch.tensor, torch.tensor]:
        """
        Calculate attention score
        Params:
            q: query (B, N, d_model)
            k: key (B, N, d_model)
            v: value (B, N, d_model)

        Returns:
            attn_output (B, N, d_model)
            attn_output_weights (B, N, N) 
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(q.size(-1))) # (B, N, d_model).(B, d_model, N) -> (B, N, N) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # (B, N, N)
        attn_output_weights = F.softmax(scores, dim = -1) # (B, N, N)
        attn_output = torch.matmul(attn_output_weights, v) # (B, N, N).(B, N, d_model) -> (B, N, d_model)
        return attn_output, attn_output_weights
    
    def splitting_heads(self, x) -> torch.tensor:
        """
        splitting item to heads
        Params:
            x: input (B, N, d_model)
        Returns:
            q: query (B, num_heads, N, d_model/num_heads)
            k: key (B, num_heads, N, d_model/num_heads)
            v: value (B, num_heads, N, d_model/num_heads)
        """
        B, N, d_model = x.shape
        hd_v = d_model // self.num_heads
        x = x.view(B, N, self.num_heads, hd_v).transpose(1, 2).contiguous()
        return x
    
    def forward(self, q, k, v, mask=None) -> torch.tensor:
        """
        calculate multi head attention
        Params:
            q: query (B, N, d_model)
            k: key (B, N, d_model)
            v: value (B, N, d_model)
            mask: mask (B, N, N)
        Returns:
            attn_output (B, N, d_model)
            attn_output_weights (B, N, N)
        """
        B, N, d_model = q.shape
        qw = self.wq(q)
        kw = self.wk(k)
        vw = self.wv(v)

        heads_q = self.splitting_heads(qw)
        heads_k = self.splitting_heads(kw)
        heads_v = self.splitting_heads(vw)

        attn_output, _ = self.scaled_dot_product_attention(heads_q, heads_k, heads_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, d_model)
        attn_output = self.wo(attn_output)
        return attn_output

