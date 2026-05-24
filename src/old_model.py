import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import time
from joblib import Parallel, delayed


def point_on_heading(point, aircraft_heading, aircraft_speed, time, wind_direction=0, wind_speed=0):

    """
    Compute the position an aircraft will finish in when flown for a set time
    """

    aircraft_angle = math.radians(90 - aircraft_heading)
    wind_angle = - math.radians(wind_direction + 90)

    aircraft_vy = aircraft_speed * math.sin(aircraft_angle)
    aircraft_vx = aircraft_speed * math.cos(aircraft_angle)

    wind_vy = wind_speed * math.sin(wind_angle)
    wind_vx = wind_speed * math.cos(wind_angle)

    x = point[0] + (aircraft_vx + wind_vx) * time
    y = point[1] + (aircraft_vy + wind_vy) * time
    
    return np.array([x,y])



def create_octagon(centre, radius):
    """Creates a regular octagon around a centre point"""
    centre_x = centre[0]
    centre_y = centre[1]
    # 8 sides + 1 to close
    angles = np.linspace(0, 2 * np.pi, 9)

    # obtain the coordinates of vertices
    octagon_coords = [
        (
            [centre_x + radius * np.cos(angle),
            centre_y + radius * np.sin(angle)]
        )
        for angle in angles
    ]

    octagon_coords.reverse()
    
    return octagon_coords


# load the data
with open("/Users/james/Documents/Exeter/Bluebird/second/holes/headings_data.pkl", "rb") as f:
    headings_data = pickle.load(f)

# with open("/Users/james/Documents/Exeter/Bluebird/second/holes/abs_positions_data.pkl", "rb") as f:
#     abs_positions_data = pickle.load(f)

# with open("/Users/james/Documents/Exeter/Bluebird/second/holes/targets_data.pkl", "rb") as f:
#     targets_data = pickle.load(f)

with open("/Users/james/Documents/Exeter/Bluebird/second/holes/rel_positions_data.pkl", "rb") as f:
    rel_positions_data = pickle.load(f)

# with open("/Users/james/Documents/Exeter/Bluebird/second/holes/abs_traffic_data.pkl", "rb") as f:
#     abs_traffic_data = pickle.load(f)

with open("/Users/james/Documents/Exeter/Bluebird/second/holes/rel_traffic_data.pkl", "rb") as f:
    rel_traffic_data = pickle.load(f)




# replace the inf values with 0

replace_dict = {np.inf: 0}

for key, value in replace_dict.items():
    rel_positions_data[rel_positions_data == key] = value
    rel_traffic_data[rel_traffic_data == key] = value



eps = 0.00000001
# exit = np.asarray((2.5-eps, 0))
entry = np.asarray((0, 0))


Z_boundary = np.array(((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)))

n_aircraft = 2
speed = 5
tt = 0.025
step_dist = speed * tt
# wind_speed = 0
# wind_direction = 0

#sequences to process in each batch
batch_size = 64
# max context length
context_length = 32
# steps between loss evaluation
eval_interval = 200
# max training steps
max_iters = 5000
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
# embedding dimensions
n_embed = 128
# number of attention heads
n_head = 8
num_kv_groups = 4
qk_norm = True
rope_base = 1_000_000.0
# layers in the network
n_layer = 8
# dimension of hidden layers in network
hidden_dim = 4*n_embed
# dropout rate
drop_rate = 0

torch.manual_seed(12)



rel_traffic_data = rel_traffic_data.reshape(len(headings_data), context_length + 1, 2*n_aircraft)

print(f"headings_data: {headings_data.shape}")
print(f"rel_positions_data: {rel_positions_data.shape}")
print(f"rel_traffic_data: {rel_traffic_data.shape}")




class HeadingTokeniser:

    def __init__(self, vocab):
        self.s_to_i = vocab
        self.i_to_s = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        ids = np.vectorize(self.s_to_i.__getitem__)(text)
        return ids

    def decode(self, ids):
        text = np.vectorize(self.i_to_s.__getitem__)(ids)
        return text


# construct vocab dictionary (note: 0 maps to 0, -1 maps to 37)
heading_options = [i*10 for i in range(1,37)]
heading_options.insert(0,0)
heading_options.append(-1)
voc = {heading_options[i] : i for i in range(len(heading_options))}
tokeniser = HeadingTokeniser(voc)
vocab_size = len(voc)

# encode and split the data into training and validation
headings_data = torch.tensor(tokeniser.encode(headings_data), dtype=torch.long)
n = int(0.9*len(headings_data))


train_data = {
    "heading" : headings_data[:n],
    "position" : torch.tensor(rel_positions_data[:n], dtype=torch.float32),
    "traffic" : torch.tensor(rel_traffic_data[:n], dtype=torch.float32)
}

val_data = {
    "heading" : headings_data[n:],
    "position" : torch.tensor(rel_positions_data[n:], dtype=torch.float32),
    "traffic" : torch.tensor(rel_traffic_data[n:], dtype=torch.float32)
}





def get_batch(split):
    """ 
    Generate a batch of data x (inputs), y (targets)
    """

    match split:
        case "train":
            data = train_data
        case "validate":
            data = val_data
    
    # make a tensor with batch_size random numbers (to pick out example trajectories)
    ix = torch.randint(len(data["heading"]), (batch_size,))
    batch = {}

    # get the data for headings, positions or targets
    # for each, build lists with the inputs (x) and the targets (y)
    for key in data.keys():
        
        dataset = data[key]

        if key == "heading":
            x = torch.tensor(np.asarray([dataset[i][:context_length] for i in ix]), dtype=torch.long)
            y = torch.tensor(np.asarray([dataset[i][1:context_length+1] for i in ix]), dtype=torch.long)
        else:
            x = torch.tensor(np.asarray([dataset[i][:context_length] for i in ix]), dtype=torch.float32)
            y = torch.tensor(np.asarray([dataset[i][1:context_length+1] for i in ix]), dtype=torch.float32)
        
        batch[key] = (x,y)

    return batch





# disable gradient calculations
@torch.no_grad()
def estimate_loss():

    out = {}

    # put the model into evaluation mode
    model.eval()

    for split in ["train", "validate"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):

            batch_dict = get_batch(split)
            X_heading, Y_heading = batch_dict["heading"]
            X_position, _ = batch_dict["position"]
            X_traffic, _ = batch_dict["traffic"]

            X_heading = X_heading.to(device)
            X_position = X_position.to(device)
            X_traffic = X_traffic.to(device)
            Y_heading = Y_heading.to(device)

            # evaluate the logits, loss
            _, loss = model(X_heading, X_position, X_traffic, Y_heading)

            losses[k] = loss.item()

        out[split] = losses.mean()
    
    # put the model back into training mode
    model.train()

    return out





class FeedForward(nn.Module):
    def __init__(self, n_embed, hidden_dim):
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
    def __init__(self):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in = n_embed,
            n_head = n_head,
            head_size = n_embed // n_head,
            num_kv_groups = num_kv_groups,
            qk_norm = qk_norm,
            dtype=torch.float32
        )
        self.ff = FeedForward(n_embed, hidden_dim)
        self.norm1 = RMSNorm(n_embed, eps=1e-6)
        self.norm2 = RMSNorm(n_embed, eps=1e-6)

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
    def __init__(self):
        super().__init__()

        self.heading_embedding_table = nn.Embedding(vocab_size, n_embed, dtype=torch.float32)

        self.final_norm = RMSNorm(n_embed)
        self.out_head = nn.Linear(n_embed, vocab_size, bias=False, dtype=torch.float32)

        cos, sin = compute_rope_params(
            head_size = n_embed // n_head,
            theta_base = rope_base,
            context_length = context_length
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [Block() for _ in range(n_layer)]
        )

        # (2,1) -> (n_embed,1)
        self.rel_spatial_embedding = nn.Linear(2, n_embed, dtype=torch.float32)
        # (2*n_aircraft,1) -> (n_embed,1)
        self.rel_traffic_embedding = nn.Linear(2*n_aircraft, n_embed, dtype=torch.float32)




    def forward(self, headx, posx, trafx, targets=None):

        # B, T = headx.shape
        
        heading_emb = self.heading_embedding_table(headx)

        rel_spatial_emb = self.rel_spatial_embedding(posx)
        rel_traffic_emb = self.rel_traffic_embedding(trafx)

        # sum the embeddings of heading, relative position, relative traffic
        x = heading_emb + rel_spatial_emb + rel_traffic_emb

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
            loss = F.cross_entropy(logits, targets)

        return logits, loss






# training loop

start_time = time.time()

model = Model()
model = model.to(device)
optimiser = torch.optim.AdamW(model.parameters())

# to keep track of losses with each training step
train_losses, validate_losses, steps = [], [], []

# initial and peak learning rates
initial_lr = 0.0001
peak_lr = 0.01
# 20% of steps for warmup
warmup_steps = int(0.2 * max_iters)
# min learning rate 10% of initial
min_lr = 0.1 * initial_lr
track_lrs = []
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1


for iter in range(max_iters):
    # if iter is one of the eval intervals, estimate and print the losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, validate loss {losses['validate']:.4f}")
        # record the losses for plotting
        steps.append(iter)
        train_losses.append(losses["train"])
        validate_losses.append(losses["validate"])
    
    batch_dict = get_batch("train")
    x_heading, y_heading = batch_dict["heading"]
    x_position, _ = batch_dict["position"]
    x_traffic, _ = batch_dict["traffic"]

    x_heading = x_heading.to(device)
    x_position = x_position.to(device)
    x_traffic = x_traffic.to(device)
    y_heading = y_heading.to(device)

    # evaluate the loss
    logits, loss = model(x_heading, x_position, x_traffic, y_heading)
    optimiser.zero_grad(set_to_none=True)
    global_step += 1
        
    # adjust learning rate based on current phase (warmup or cosine annealing)
    if global_step < warmup_steps:
        # linear warmup
        lr = initial_lr + global_step * lr_increment  
    else:
        # cosine annealing after warmup
        progress = ((global_step - warmup_steps) / (max_iters - warmup_steps))
        lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
    # apply the learning rate to the optimiser
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr
    track_lrs.append(optimiser.param_groups[0]["lr"])
    
    loss.backward()

    # gradient clipping after warm up
    if global_step >= warmup_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimiser.step()


end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"training completed in {execution_time_minutes:.1f} minutes")


