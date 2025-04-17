import torch
import torch.nn as nn

from .transfer_model import TransferModel


class SixTokenDecoderTransferModel(TransferModel):

    # TODO - consider just passing the tokenizer?
    def __init__(self, pos1_vocab_size, pos2_vocab_size, pos3_vocab_size,
                 vel1_vocab_size, vel2_vocab_size, vel3_vocab_size,
                 d_model=256, n_heads=4, n_layers=4, d_ff=1024, dropout=0.1, max_seq_len=512):
        """
        A transformer decoder model for six-token sequences: three for position, three for velocity.

        Args:
            pos1_vocab_size (int): Vocabulary size for position token 1 (e.g., r or x).
            pos2_vocab_size (int): Vocabulary size for position token 2 (e.g., theta or y).
            pos3_vocab_size (int): Vocabulary size for position token 3 (e.g., phi or z).
            vel1_vocab_size (int): Vocabulary size for velocity token 1 (e.g., vr or vx).
            vel2_vocab_size (int): Vocabulary size for velocity token 2 (e.g., vtheta or vy).
            vel3_vocab_size (int): Vocabulary size for velocity token 3 (e.g., vphi or vz).
            d_model (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of decoder layers.
            d_ff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
            max_seq_len (int): Maximum sequence length for positional embeddings.
        """
        super().__init__()
        self.pos1_vocab_size = pos1_vocab_size
        self.pos2_vocab_size = pos2_vocab_size
        self.pos3_vocab_size = pos3_vocab_size
        self.vel1_vocab_size = vel1_vocab_size
        self.vel2_vocab_size = vel2_vocab_size
        self.vel3_vocab_size = vel3_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        # Embeddings for each token type
        self.pos1_emb = nn.Embedding(pos1_vocab_size, d_model)
        self.pos2_emb = nn.Embedding(pos2_vocab_size, d_model)
        self.pos3_emb = nn.Embedding(pos3_vocab_size, d_model)
        self.vel1_emb = nn.Embedding(vel1_vocab_size, d_model)
        self.vel2_emb = nn.Embedding(vel2_vocab_size, d_model)
        self.vel3_emb = nn.Embedding(vel3_vocab_size, d_model)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        self.drop = nn.Dropout(dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads for each token type
        self.pos1_head = nn.Linear(d_model, pos1_vocab_size)
        self.pos2_head = nn.Linear(d_model, pos2_vocab_size)
        self.pos3_head = nn.Linear(d_model, pos3_vocab_size)
        self.vel1_head = nn.Linear(d_model, vel1_vocab_size)
        self.vel2_head = nn.Linear(d_model, vel2_vocab_size)
        self.vel3_head = nn.Linear(d_model, vel3_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for embeddings and output heads."""
        for emb in [self.pos1_emb, self.pos2_emb, self.pos3_emb, self.vel1_emb, self.vel2_emb, self.vel3_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for head in [self.pos1_head, self.pos2_head, self.pos3_head, self.vel1_head, self.vel2_head, self.vel3_head]:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

    def forward(self, pos1_tokens, pos2_tokens, pos3_tokens, vel1_tokens, vel2_tokens, vel3_tokens):
        """
        Forward pass of the model.

        Args:
            pos1_tokens (torch.Tensor): Position token 1 sequence, shape (B, T).
            pos2_tokens (torch.Tensor): Position token 2 sequence, shape (B, T).
            pos3_tokens (torch.Tensor): Position token 3 sequence, shape (B, T).
            vel1_tokens (torch.Tensor): Velocity token 1 sequence, shape (B, T).
            vel2_tokens (torch.Tensor): Velocity token 2 sequence, shape (B, T).
            vel3_tokens (torch.Tensor): Velocity token 3 sequence, shape (B, T).

        Returns:
            tuple: (pos1_logits, pos2_logits, pos3_logits, vel1_logits, vel2_logits, vel3_logits),
                   each of shape (B, T, vocab_size).
        """
        B, T = pos1_tokens.shape

        # Embed each token sequence
        pos1_e = self.pos1_emb(pos1_tokens)  # (B, T, d_model)
        pos2_e = self.pos2_emb(pos2_tokens)
        pos3_e = self.pos3_emb(pos3_tokens)
        vel1_e = self.vel1_emb(vel1_tokens)
        vel2_e = self.vel2_emb(vel2_tokens)
        vel3_e = self.vel3_emb(vel3_tokens)

        # Sum the embeddings
        x = pos1_e + pos2_e + pos3_e + vel1_e + vel2_e + vel3_e

        # Add positional encoding
        pos_slice = self.pos_emb[:, :T, :]
        x = x + pos_slice
        x = self.drop(x)  # (B, T, d_model)

        # Prepare for transformer: (T, B, d_model)
        x = x.permute(1, 0, 2)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        # Dummy memory for decoder
        memory = x.new_zeros((1, B, self.d_model))

        # Decoder
        out = self.decoder(x, memory, tgt_mask=causal_mask)  # (T, B, d_model)

        # Back to (B, T, d_model)
        out = out.permute(1, 0, 2)

        # Apply output heads
        pos1_logits = self.pos1_head(out)
        pos2_logits = self.pos2_head(out)
        pos3_logits = self.pos3_head(out)
        vel1_logits = self.vel1_head(out)
        vel2_logits = self.vel2_head(out)
        vel3_logits = self.vel3_head(out)

        return pos1_logits, pos2_logits, pos3_logits, vel1_logits, vel2_logits, vel3_logits

    def to_dict(self):
        """Serialize model parameters to a dictionary."""
        return {
            "class_name": "SixTokenDecoderTransferModel",
            "pos1_vocab_size": self.pos1_vocab_size,
            "pos2_vocab_size": self.pos2_vocab_size,
            "pos3_vocab_size": self.pos3_vocab_size,
            "vel1_vocab_size": self.vel1_vocab_size,
            "vel2_vocab_size": self.vel2_vocab_size,
            "vel3_vocab_size": self.vel3_vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len
        }
    