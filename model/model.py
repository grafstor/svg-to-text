import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
    
        pos_encoding = torch.zeros(max_len, dim)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0)) / dim) # 1000^(2i/dim_model)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout_p):
        super().__init__()
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.act = nn.SiLU()

        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.dropout(x)
        x = self.act(self.conv2(x))
        x = self.dropout(x)

        x = self.pool(x)
        
        return x
        
class ConvEncoder(nn.Module):
    def __init__(self, dim, seq_len, dropout_p=0.01):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len

        blocks = []
        dims = [11, dim, dim*2, dim*4]

        for i in range(len(dims)-1):
            block = ConvBlock(
                dim_in=dims[i],
                dim_out=dims[i+1],
                dropout_p=dropout_p
            )
            blocks.append(block)
            
        self.blocks = nn.ModuleList(blocks)
        
        out_dim = (seq_len//(2**(len(dims)-1))) * dims[-1]

        self.out = nn.Linear(out_dim, dim)

    def forward(self, x):

        x = x.permute(0, 2, 1) # Reshape for 1D conv

        for block in self.blocks:
            x = block(x)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        x = self.out(x)

        x = x.unsqueeze(1)
        return x


class STTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        seq_len,
        n_heads,
        n_encoder_layers,
        n_decoder_layers,
        dropout_p,
        voc=None,
    ):
        super().__init__()

        self.dim = dim
        self.voc = voc
        
        self.src_encoder = ConvEncoder(
            dim=dim,
            seq_len=seq_len
        )
        
        self.positional_encoder = PositionalEncoding(
            dim=dim, dropout_p=dropout_p, max_len=1000
        )
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=n_heads,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout_p
        )
        
        self.out = nn.Linear(dim, vocab_size)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        
        src = self.src_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.dim)
        
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        transformer_out = self.transformer(
            src, tgt, 
            tgt_mask=tgt_mask, 
            src_key_padding_mask=src_pad_mask, 
            tgt_key_padding_mask=tgt_pad_mask
        )
    
        out = self.out(transformer_out)
        
        return out
