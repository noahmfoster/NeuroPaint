import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config_utils import DictConfig
from typing import Optional
from utils.pos_embed import get_cos_sin, apply_rotary_pos_emb
from transformers.activations import ACT2FN


##############blocks###############
# MLP
class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()
        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))
    
# Attention module.
class NeuralAttention(nn.Module):

    def __init__(self, idx, hidden_size, n_heads, use_bias,  max_F, dropout, use_rope=True):
        super().__init__()
        
        self.idx = idx

        # Architecture config
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, f"Hidden dim is not multiple of head size"
        self.head_size = self.hidden_size // self.n_heads
        self.use_rope = use_rope

        # Attention parameters
        self.query = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.value  = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.attn_dropout = dropout

        # Final projection
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # RoPE parameters
        if use_rope:
            cos, sin = get_cos_sin(self.head_size, max_F, base=10000., dtype=self.query.weight.dtype, device=self.query.weight.device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)


    def forward(
        self,       
        x:          torch.FloatTensor,                      # (bs, seq_len, hidden_size)  
        timestamp:  Optional[torch.LongTensor] = None,      # (bs, seq_len) 
    ) -> torch.FloatTensor:                                 # (bs, seq_len, hidden_size)

        B, T, _  = x.size()     # batch size and fea len

        # Compute query, key, value for attention
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)        #(B,n_heads,T,head_size)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)      #(B,n_heads,T,head_size)

        # Apply rotations to encode relative positions
        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)  # (B,n_heads,T,head_size)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=(self.attn_dropout if self.training else 0.0), is_causal=False) # (B,n_heads,T,head_size)
        out = out.transpose(1, 2).contiguous().view(B,T, self.hidden_size)       # (B, T, hidden_size)

        return self.out_proj(self.dropout(out)) # (B, T, hidden_size)

# Transformer Block layer: bidirectional self-attention + mlp
class Block(nn.Module):
    
    def __init__(self, idx, max_F, config: DictConfig):
        super().__init__()

        self.idx = idx
        # Architecture config
        self.use_rope = config.use_rope

        # Encoder block
        self.ln1 = nn.LayerNorm(config.hidden_size) 
        self.attn = NeuralAttention(idx, config.hidden_size, config.n_heads, config.attention_bias, max_F, config.dropout, config.use_rope)
        self.ln2 = nn.LayerNorm(config.hidden_size) 
        self.mlp = NeuralMLP(config.hidden_size, config.inter_size, config.act, config.mlp_bias, config.dropout)

        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self, 
        x:          torch.FloatTensor,                  # (bs, seq_len, hidden_size)
        timestamp:  Optional[torch.LongTensor] = None,  # (bs, seq_len) #tdb this need to be the kept timestamps after masking     
    ) -> torch.FloatTensor :                            # (bs, seq_len, hidden_size)
        
        # LN -> Attention -> Residual connectiob
        x = x + self.attn(self.ln1(x), timestamp if self.use_rope else None)

        # LN -> MLP -> Residual connection
        x = x + self.mlp(self.ln2(x))

        return x

    def fixup_initialization(self, n_layers):
        temp_state_dic = {}
        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * param
            elif name.endswith("value.weight"):
                temp_state_dic[name] = (0.67 * (n_layers) ** (- 1. / 4.)) * (param * (2**0.5))

        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)   
######################end of block#########################