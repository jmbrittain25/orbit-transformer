import torch
import torch.nn as nn


class ThreeTokenGPTConfig:
    def __init__(
        self,
        r_vocab_size,
        theta_vocab_size,
        phi_vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=512
    ):
        self.r_vocab_size = r_vocab_size
        self.theta_vocab_size = theta_vocab_size
        self.phi_vocab_size = phi_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len


class ThreeTokenGPTDecoder(nn.Module):
    """
    Decoder-only transformer where each time step is represented by 3 discrete tokens: (r, theta, phi).
    We sum their embeddings + positional embedding to get a single d_model vector per time step.
    """

    def __init__(self, config: ThreeTokenGPTConfig):
        super().__init__()
        self.config = config

        # Separate embeddings
        self.r_emb = nn.Embedding(config.r_vocab_size, config.d_model)
        self.theta_emb = nn.Embedding(config.theta_vocab_size, config.d_model)
        self.phi_emb = nn.Embedding(config.phi_vocab_size, config.d_model)

        # Positional embedding (learnable)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))

        self.drop = nn.Dropout(config.dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)

        # Final output: we might want 3 separate predictions (one for r, one for theta, one for phi),
        # or unify them into one big classification step. 
        # Here we demonstrate a single output for each dimension:
        self.r_head = nn.Linear(config.d_model, config.r_vocab_size)
        self.theta_head = nn.Linear(config.d_model, config.theta_vocab_size)
        self.phi_head = nn.Linear(config.d_model, config.phi_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for emb in [self.r_emb, self.theta_emb, self.phi_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        nn.init.normal_(self.r_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.theta_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.phi_head.weight, mean=0.0, std=0.02)
        # pos_emb is zero-initialized

    def forward(self, r_tokens, theta_tokens, phi_tokens):
        """
        r_tokens, theta_tokens, phi_tokens : shape (B, T)
        Returns:
          r_logits, theta_logits, phi_logits : shape (B, T, r_vocab), etc.
        """
        B, T = r_tokens.shape

        # Embeddings
        r_e = self.r_emb(r_tokens)         # (B, T, d_model)
        th_e = self.theta_emb(theta_tokens)
        ph_e = self.phi_emb(phi_tokens)

        x = r_e + th_e + ph_e  # sum dimension embeddings

        # Add positional
        pos_slice = self.pos_emb[:, :T, :]  # (1, T, d_model)
        x = x + pos_slice
        x = self.drop(x)  # (B, T, d_model)

        # Transform to (T, B, d_model) for nn.Transformer
        x = x.permute(1, 0, 2)  # (T, B, d_model)

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))

        # Create a dummy memory with correct shape for decoder
        memory = x.new_zeros((1, B, self.config.d_model))  # (1, B, d_model)
        
        out = self.decoder(
            x,                      # (T, B, d_model)
            memory=memory,          # (1, B, d_model)
            tgt_mask=causal_mask
        )  # (T, B, d_model)


        # Back to (B, T, d_model)
        out = out.permute(1, 0, 2)

        # Separate heads
        r_logits = self.r_head(out)        # (B, T, r_vocab_size)
        theta_logits = self.theta_head(out)# (B, T, theta_vocab_size)
        phi_logits = self.phi_head(out)    # (B, T, phi_vocab_size)

        return r_logits, theta_logits, phi_logits
    