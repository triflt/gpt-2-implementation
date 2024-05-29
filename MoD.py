import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import FeedFoward

batch_size = 32
block_size = 4
max_iters = 6000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 768
n_head = 6
n_layer = 6
dropout = 0.2

class MoDTransformerBlock(nn.Module):
    """Wrapper class for integrating a transformer block with Mixture-of-Depths routing.

    Attributes:
        transformer_block (nn.Module): Transformer block to be wrapped.
        router_mlp (nn.Linear): MLP layer for calculating router weights.
        aux_mlp (nn.Linear): MLP layer for calculating auxiliary routing decision.
        capacity (float): Capacity of the mixture-of-depths routing. Default is 0.125.
        aux_loss (torch.Tensor): Auxiliary loss for training auxiliary MLP.

    Notes:
        MoD Paper Link: https://arxiv.org/pdf/2404.02258
    """

    def __init__(
        self,
        transformer_block,
        hidden_size: int,
        capacity: float = 0.125,
    ):
        """Initialize the MoD wrapped transformer block.

        Args:
            transformer_block (...): Transformer block to be wrapped.
            hidden_size (int): Hidden size of the transformer block.
            capacity (float, optional): Capacity of the mixture-of-depths routing.
                Defaults to 0.125.

        Raises:
            ValueError: If the capacity is not in the range (0, 1].

        Note:
            The default capacity of 0.125 is according to the original paper.
        """
        super(MoDTransformerBlock, self).__init__()
        self.transformer_block = transformer_block
        self.router_mlp: nn.Linear = nn.Linear(hidden_size, 1)
        self.aux_mlp: nn.Linear = nn.Linear(hidden_size, 1)

        if capacity <= 0 or capacity > 1:
            raise ValueError(
                f"Capacity must be in the range (0, 1]. Got: {capacity}"
            )
        self.capacity = capacity

    def forward(self, x: torch.Tensor):
        """Forward pass through the MoD wrapped transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        B, sequence_length, _ = x.shape

        # Calculate scalar router weights logits.
        # [Shape] router_weights: (batch_size, sequence_length)
        router_weights: torch.Tensor = self.router_mlp(x).squeeze(-1)

        if self.training:
            # ┌────────────────────────────────────────────────────────┐
            # │                  > Training STAGE <                    │
            # └────────────────────────────────────────────────────────┘

            # Calculate top-k indices based on router weights.
            k = int(sequence_length * self.capacity)

            # [Shape] topk_indices: (batch_size, k)
            topk_indices = torch.topk(router_weights, k, dim=-1).indices

            # Generate binary labels for auxiliary MLP training.
            # [Shape] aux_targets: (batch_size, sequence_length)
            aux_targets = torch.zeros_like(router_weights)
            aux_targets.scatter_(1, topk_indices, 1.0)

            # Calculate auxiliary logits for training auxiliary MLP.
            # Stop gradient flow to the auxiliary decision. (means `detach()`)
            # [Shape] aux_logits: (batch_size, sequence_length)
            aux_logits = self.aux_mlp(x.detach()).squeeze(-1)

            # Calculate auxiliary routing decision (binary 0/1 index).
            # [Shape] aux_decision: (batch_size, sequence_length)
            aux_decision = (torch.sigmoid(aux_logits) > 0.5).float()

            # Calculate auxiliary loss (Binary Cross Entropy) and save for backward pass.
            self.aux_loss = F.binary_cross_entropy_with_logits(
                aux_logits, aux_targets
            )
        else:
            # ┌────────────────────────────────────────────────────────┐
            # │                 > Inference STAGE <                    │
            # └────────────────────────────────────────────────────────┘

            # Calculate auxiliary logits for training auxiliary MLP.
            # [Shape] aux_logits: (batch_size, sequence_length)
            aux_logits = self.aux_mlp(x.detach()).squeeze(-1)

            # Calculate auxiliary routing decision (binary 0/1 index).
            # [Shape] aux_decision: (batch_size, sequence_length)
            aux_decision = (torch.sigmoid(aux_logits) > 0.5).float()

        # Tokens not routed for specialized computation will skip it via the residual connection.
        # [Shape] output: (batch_size, sequence_length, hidden_size)
        output = x.clone()
        #print(f' output first {output.shape}')

        # Assure that the auxiliary decision is a boolean tensor.
        aux_decision = aux_decision.bool()

        for b in range(B):
            # Extract tokens and router that need to go through the transformer block.
            # [Shape] selected_tokens_emb: (selected_tokens_count, hidden_size)
            selected_tokens_emb = x[b, aux_decision[b]]
            # [Shape] selected_router_weights: (selected_tokens_count, 1)
            selected_router_weights = router_weights[
                b, aux_decision[b]
            ].unsqueeze(-1)
            
            #print(f'selected_tokens_emb.shape {selected_tokens_emb.shape}')
            #print(f'selected_router_weights.shape {selected_router_weights.shape}')
            if selected_tokens_emb.shape[0] > 0:
                # Apply the transformer block to the selected tokens.
                transformer_tokens_emb = (
                    self.transformer_block(selected_tokens_emb)
                    * selected_router_weights
                )

                # Scatter the tokens into output according to the auxiliary decision.
                output[b, aux_decision[b]] = transformer_tokens_emb
        #print(f'output.shape {output.shape}')
        return output

# Old version
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (B, T, C)
        # output of size (B, T, D)
        #B,T,C = x.shape
        T, C = x.shape
        k = self.key(x)   # (B,T,D)
        q = self.query(x) # (B,T,D)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, D) @ (B, D, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,D)
        out = wei @ v # (B, T, T) @ (B, T, D) -> (B, T, D)
        return out

class MultiHeadAttentionB(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TrasnformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionB(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #print(x.shape)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPTLanguageMoD(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[MoDTransformerBlock(TrasnformerBlock(n_embd, n_head=n_head), hidden_size=n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
