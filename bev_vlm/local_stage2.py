import torch
import torch.nn.functional as F
from torch import nn

from .connectors import MLPConnector


class LocalBEVQAStage2Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_id,
        hidden_size=256,
        num_tasks=3,
        connector_token_grid=(4, 4),
        connector_dropout=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.connector = MLPConnector(
            token_grid=connector_token_grid,
            hidden_size=hidden_size,
            dropout=connector_dropout,
        )
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.task_embedding = nn.Embedding(num_tasks, hidden_size)
        self.question_encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=connector_dropout,
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def encode_context(self, question_ids, task_ids, bev_tensors):
        question_emb = self.token_embedding(question_ids)
        _, question_hidden = self.question_encoder(question_emb)
        question_hidden = question_hidden[-1]
        task_hidden = self.task_embedding(task_ids)
        query = (question_hidden + task_hidden).unsqueeze(1)
        bev_tokens = self.connector(bev_tensors)
        attended_bev, _ = self.cross_attention(query, bev_tokens, bev_tokens)
        context = attended_bev.squeeze(1) + question_hidden + task_hidden
        return context, bev_tokens

    def forward(self, question_ids, answer_ids, task_ids, bev_tensors):
        context, bev_tokens = self.encode_context(question_ids, task_ids, bev_tensors)
        decoder_input_ids = answer_ids[:, :-1]
        target_ids = answer_ids[:, 1:]

        decoder_emb = self.token_embedding(decoder_input_ids)
        context_expanded = context.unsqueeze(1).expand(-1, decoder_emb.size(1), -1)
        decoder_input = torch.cat([decoder_emb, context_expanded], dim=-1)
        decoder_output, _ = self.decoder(decoder_input, context.unsqueeze(0))
        logits = self.output_head(decoder_output)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        return {
            "loss": loss,
            "logits": logits,
            "connector_tokens": bev_tokens,
        }

    @torch.no_grad()
    def generate(
        self,
        question_ids,
        task_ids,
        bev_tensors,
        bos_id,
        eos_id,
        max_new_tokens=128,
    ):
        context, _ = self.encode_context(question_ids, task_ids, bev_tensors)
        hidden = context.unsqueeze(0)
        current = torch.full(
            (question_ids.size(0), 1),
            bos_id,
            dtype=torch.long,
            device=question_ids.device,
        )
        generated = []
        for _ in range(max_new_tokens):
            embedded = self.token_embedding(current[:, -1:])
            decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            output, hidden = self.decoder(decoder_input, hidden)
            logits = self.output_head(output[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            current = torch.cat([current, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_id):
                break
        if not generated:
            return current[:, 1:]
        return torch.cat(generated, dim=1)
