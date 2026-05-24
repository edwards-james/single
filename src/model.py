import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, n_embed, hidden_dim, drop_rate):
        super().__init__()
        self.fc1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.fc2 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        x = self.fc3(x)
        return self.drop(x)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(n_embed))
        self.shift = nn.Parameter(torch.zeros(n_embed)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)



def compute_rope_params(head_size, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_size % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_size, 2, dtype=dtype)[: (head_size // 2)].float() / head_size))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    # [context_length, head_size // 2]
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    # Expand angles to match the head_size
    # [context_length, head_size]
    angles = torch.cat([angles, angles], dim=1)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin



def apply_rope(x, cos, sin):
    # x: [batch_size, n_head, seq_len, head_dim]
    _, _, seq_len, head_size = x.shape
    assert head_size % 2 == 0, "Head dimension must be even"

    # aplit x into first half and second half
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]

    # adjust sin and cos shapes
    # [1, 1, seq_len, head_size]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # apply rotation transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)




class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, n_head, num_kv_groups, head_size=None, qk_norm=False, dtype=None
        ):

        super().__init__()
        assert n_head % num_kv_groups == 0, "n_head must be divisible by num_kv_groups"

        # number of attention heads
        self.n_head = n_head
        # number of key-value groups
        self.num_kv_groups = num_kv_groups
        self.group_size = n_head // num_kv_groups

        if head_size is None:
            assert d_in % n_head == 0, "d_in must be divisible by n_head if head_size is not set"
            head_size = d_in // n_head

        self.head_size = head_size
        self.d_out = n_head * head_size

        # weight matrices for q, k, v
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_size, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_size, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_size, eps=1e-6)
            self.k_norm = RMSNorm(head_size, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # get q, k, v from x using the weight matrices
        queries = self.W_query(x)  # (b, num_tokens, n_head * head_size)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_size)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_size)

        # Reshape
        queries = queries.view(b, num_tokens, self.n_head, self.head_size).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_size).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_size).transpose(1, 2)

        # optional normalisation
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_size**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)
    



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in = config["n_embed"],
            n_head = config["n_head"],
            head_size = config["n_embed"] // config["n_head"],
            num_kv_groups = config["num_kv_groups"],
            qk_norm = config["qk_norm"],
            dtype=torch.float32
        )
        self.ff = FeedForward(config["n_embed"], config["hidden_dim"], config["drop_rate"])
        self.norm1 = RMSNorm(config["n_embed"], eps=1e-6)
        self.norm2 = RMSNorm(config["n_embed"], eps=1e-6)

    def forward(self, x, mask, cos, sin):

        shortcut = x
        x = self.norm1(x)
        # [batch_size, num_tokens, n_embed]
        x = self.att(x, mask, cos, sin)
        # add skip
        x = x + shortcut

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x





class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pad_token_id = config["pad_token_id"]

        self.final_norm = RMSNorm(config["n_embed"])
        self.out_head = nn.Linear(config["n_embed"], config["vocab_size"], bias=False, dtype=torch.float32)

        cos, sin = compute_rope_params(
            head_size = config["n_embed"] // config["n_head"],
            theta_base = config["rope_base"],
            context_length = config["context_length"]
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config["n_layer"])]
        )

        # the embeddings
        
        self.heading_embedding_table = nn.Embedding(config["vocab_size"], config["n_embed"], dtype=torch.float32)
        # (2,1) -> (n_embed,1)
        self.rel_spatial_embedding = nn.Linear(2, config["n_embed"], dtype=torch.float32)
        # (2*n_aircraft,1) -> (n_embed,1)
        self.rel_traffic_embedding = nn.Linear(2*config["n_aircraft"], config["n_embed"], dtype=torch.float32)

        self.input_proj = nn.Linear(3 * config["n_embed"], config["n_embed"])


    def forward(self, headx, posx, trafx, targets=None):

        heading_emb = self.heading_embedding_table(headx)
        rel_spatial_emb = self.rel_spatial_embedding(posx)
        rel_traffic_emb = self.rel_traffic_embedding(trafx)

        # concat the embeddings of heading, relative position, relative traffic -> x is now (B, T, 3 * n_embed)
        x = torch.cat([heading_emb, rel_spatial_emb, rel_traffic_emb], dim=-1)
        x = self.input_proj(x)  # (B, T, n_embed)

        num_tokens = x.shape[1]
        # upper triangular matrix mask (causal attn)
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)
        
        for block in self.blocks:
            x = block(x, mask, self.cos, self.sin)
        
        x = self.final_norm(x)

        logits = self.out_head(x.to(torch.float32))

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.pad_token_id)

        return logits, loss


