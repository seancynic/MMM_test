import torch
import torch.nn as nn
from transformers import BertModel


class MotionEncoder(nn.Module):
    def __init__(self, vqvae, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        # VQVAE for motion encoding
        self.vqvae = vqvae
        self.learn_tok_emb = nn.Embedding(3, self.vqvae.vqvae.code_dim)  # 3 = [end_id, blank_id, mask_id]

        # Projection
        self.proj = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, motion_ids, mask=None):
        not_learnt_motion_ids = motion_ids < self.vqvae.vqvae.num_code
        learnt_motion_ids = ~not_learnt_motion_ids

        motion_embeds = torch.empty((*motion_ids.shape, self.vqvae.vqvae.code_dim), device=motion_ids.device)
        motion_embeds[not_learnt_motion_ids] = self.vqvae.vqvae.quantizer.dequantize(motion_ids[not_learnt_motion_ids]).requires_grad_(False)
        motion_embeds[learnt_motion_ids] = self.learn_tok_emb(motion_ids[learnt_motion_ids] - self.vqvae.vqvae.num_code)

        motion_embeds = self.proj(motion_embeds)  # (batch, max_m, embed_dim)
        motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=mask)
        motion_embeds = self.norm(motion_embeds)  # (batch, max_m, embed_dim)

        return motion_embeds


class MotionDecoder(nn.Module):
    def __init__(self, vocab_m, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        # Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,   # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Projection
        self.proj = nn.Linear(embed_dim, vocab_m)

    def forward(self, motion_embeds, mask=None):
        motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=mask)
        motion_embeds = self.norm(motion_embeds)  # (batch, max_m, embed_dim)
        motion_logits = self.proj(motion_embeds)  # (batch, max_m, vocab_m)

        return motion_logits


class TextHead(nn.Module):
    def __init__(self, embed_dim, vocab_t):
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_t)

    def forward(self, text_embeds):
        text_logits = self.proj(text_embeds)
        return text_logits

    # def __init__(self, embed_dim, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0.1):
    #     super().__init__()
    #     # Decoder
    #     encoder_layer = nn.TransformerEncoderLayer(
    #         d_model=embed_dim,
    #         nhead=num_heads,
    #         dim_feedforward=int(embed_dim * mlp_ratio),
    #         dropout=dropout,
    #         activation='gelu',
    #         batch_first=True,
    #         norm_first=True,   # Pre-LayerNorm
    #     )
    #     self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    #     self.norm = nn.LayerNorm(embed_dim)
    #
    #     # Projection
    #     self.proj = nn.Linear(self.vqvae.vqvae.code_dim, embed_dim)
    #
    # def forward(self, motion_embeds, mask=None):
    #     motion_embeds = self.encoder(motion_embeds, src_key_padding_mask=mask)
    #     motion_embeds = self.norm(motion_embeds)
    #     motion_embeds = self.proj(motion_embeds)
    #
    #     return motion_embeds


class BiTMBERT(nn.Module):
    def __init__(self, bert_name, vqvae, vocab_m, max_t, max_m, first_modality, dropout_rate):
        super().__init__()
        # Backbone
        self.bert = BertModel.from_pretrained(bert_name)
        # Text Head
        self.text_head = TextHead(self.bert.config.hidden_size, self.bert.config.vocab_size)
        # Motion Encoder and Decoder
        self.motion_encoder = MotionEncoder(vqvae, self.bert.config.hidden_size, dropout=dropout_rate)
        self.motion_decoder = MotionDecoder(vocab_m, self.bert.config.hidden_size, dropout=dropout_rate)

        self.max_t = max_t
        self.max_m = max_m
        self.fm = first_modality

    def forward(self, text_ids, motion_ids, text_mask, motion_mask):
        # Get text and motion embeddings
        text_embeds = self.bert.embeddings.word_embeddings(text_ids)  # (batch, max_t, hidden_size)
        motion_embeds = self.motion_encoder(motion_ids, motion_mask)  # (batch, max_m, hidden_size)

        # Concatenate text and motion embeddings
        if self.fm == 'motion':
            combined_embeds = torch.cat([motion_embeds, text_embeds], dim=1)  # (batch, max_m + max_t, hidden_size)
            combined_mask = torch.cat([motion_mask, text_mask], dim=1)        # (batch, max_m + max_t)
        elif self.fm == 'text':
            combined_embeds = torch.cat([text_embeds, motion_embeds], dim=1)  # (batch, max_t + max_m, hidden_size)
            combined_mask = torch.cat([text_mask, motion_mask], dim=1)        # (batch, max_t + max_m)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")

        # Backbone
        bert_outputs = self.bert(inputs_embeds=combined_embeds, attention_mask=combined_mask, return_dict=True)
        embeds = bert_outputs.last_hidden_state  # (batch, max + max, hidden_size)

        # Separate text and motion embeddings
        if self.fm == 'motion':
            text_embeds = embeds[:, self.max_m:]    # (batch, max_t, hidden_size)
            motion_embeds = embeds[:, :self.max_m]  # (batch, max_m, hidden_size)
        elif self.fm == 'text':
            text_embeds = embeds[:, :self.max_t]    # (batch, max_t, hidden_size)
            motion_embeds = embeds[:, self.max_t:]  # (batch, max_m, hidden_size)
        else:
            raise ValueError(f"Unknown first modality: {self.fm}")

        # Predict text and motion logits
        text_logits = self.text_head(text_embeds)           # (batch, max_t, vocab_t)
        motion_logits = self.motion_decoder(motion_embeds)  # (batch, max_m, vocab_m)

        return {
            'logits_t': text_logits,
            'logits_m': motion_logits,
        }