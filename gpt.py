import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 6
droput = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

def get_batch(split):

    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((block_size,block_size))))

        self.dropout = nn.Dropout(droput)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B, 8, 32
        q = self.query(x) # B, 8, 32
        #compute attention scores affinities
        # B, 8, 32 @ B, 32, 8 --> B, 8, 8
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim=-1) #(B,T,T)
        wei = self.dropout(wei)
        v = self.value(x) # B, 8, 32
        out = wei @ v # B, 8, 8  @  B, 8, 32  
        return out # B, 8, 32

# Forward Pass in MultiHeadAttention

# 	1.	Input Tensor Shape:
# 	•	Suppose the input tensor x has shape (B, T, C), where B = 4, T = 8, and C = 32.
# 	2.	Process Each Head:
# 	•	Each Head processes x independently and outputs a tensor of shape (B, T, head_size).
# 	•	With head_size = 8, each head will produce (4, 8, 8).
# 	3.	Concatenation:
# 	•	After all heads have processed the input, the results are concatenated along the last dimension.
# 	•	Since there are 4 heads and each produces an output of size 8, the concatenated output will be:
# 	•	Shape: (B, T, num_heads * head_size) = (4, 8, 4 * 8) = (4, 8, 32).    

# 1.	Initialize MultiHeadAttention:
# self.sa_heads = MultiHeadAttention(4, 32 // 4)  # num_heads=4, head_size=8
# 2.	Input Tensor x:
# •	Shape: (4, 8, 32)
# 3.	Forward Pass:
# •	Each Head processes x and outputs tensors of shape (4, 8, 8).
# •	Concatenate the outputs of all 4 heads along the last dimension:
# •	Resulting shape: (4, 8, 32)
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(droput)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# In original transformer paper the inner layer of Feed forward iha a multiplication of 4 the dimension =512 and the feedforward 
# neural network inner layer =2048 i.e why we also multiplied with 4
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(droput)
        )

    def forward(self, x):
        return self.net(x)


# Why Are Residual Connections Used?

# 	1.	Preventing Vanishing Gradients: As models get deeper (i.e., more layers), gradients can vanish, leading to ineffective learning. By adding the input directly to the output of a layer, residual connections help gradients flow through the network, improving training.
# 	2.	Helping with Gradient Flow: Residual connections allow gradients to bypass certain layers during backpropagation, which helps stabilize the learning process.
# 	3.	Improving Convergence: Networks with residual connections can converge faster, allowing them to learn complex patterns more efficiently.
# 	•	First Addition:
# 	•	x = x + self.sa(x):
# 	•	The output of the multi-head attention (self.sa(x)) is added to the original input x.
# 	•	This ensures that the attention mechanism modifies the input while retaining some of the original information.
# 	•	Second Addition:
# 	•	x = x + self.ffwd(x):
# 	•	The output of the feedforward network (self.ffwd(x)) is added to the result from the previous step.
# 	•	Again, this retains some of the original information while incorporating the new learned features from the feedforward network.

# How It Helps

# 	•	The model can learn both transformations (through self.sa(x) and self.ffwd(x)) and retain the original input information.
# 	•	This approach enables the network to maintain essential input details that might otherwise be lost during deep transformations, allowing deeper networks without suffering from degradation problems.

# Example of Residual Path in Action

# In simpler terms:

# 	•	If x originally carries important information, the attention and feedforward operations should enhance this information.
# 	•	By adding x back after these operations, the model ensures that it doesn’t lose track of key features present in the original input.


class Block(nn.Module):
    """Transformer block:  communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_tabe = nn.Embedding(block_size, n_embd)   
        # self.sa_head = Head(n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd//4) # i.e 4 heads of 8-dimmensional self-attention   
        # self.ffwd = FeedForward(n_embd)  
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(in_features=n_embd, out_features=vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_tabe(torch.arange(T, device=device))
        x = token_emb + pos_emb
        # x = self.sa_head(x) # apply self attention head 
        # x = self.sa_heads(x)
        # x = self.ffwd(x) # ffwd is added instead of directly going for logits give some time to think what should we do with the info we found in head
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # since now we are using positional embedding our idx can't be greater than block_size because our 
            # positional embedding has just embedding upto block_size

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
