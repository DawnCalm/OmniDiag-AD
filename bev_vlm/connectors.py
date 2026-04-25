from collections import OrderedDict

import torch
from torch import nn


def count_trainable_parameters(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


class _BaseConnector(nn.Module):
    def __init__(self, modalities=("camera", "lidar", "fused"), token_grid=(4, 4)):
        super().__init__()
        self.modalities = tuple(modalities)
        self.token_grid = tuple(token_grid)

    def _pool_modality_tensor(self, tensor):
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(tensor.shape)}")
        pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, self.token_grid)
        return pooled.flatten(2).transpose(1, 2).contiguous()


class MLPConnector(_BaseConnector):
    def __init__(
        self,
        modalities=("camera", "lidar", "fused"),
        token_grid=(4, 4),
        hidden_size=256,
        dropout=0.1,
    ):
        super().__init__(modalities=modalities, token_grid=token_grid)
        self.hidden_size = hidden_size
        self.projections = nn.ModuleDict()
        self.modality_embeddings = nn.ParameterDict()
        for modality in self.modalities:
            self.projections[modality] = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.modality_embeddings[modality] = nn.Parameter(
                torch.zeros(1, self.token_grid[0] * self.token_grid[1], hidden_size)
            )

    @property
    def num_tokens_per_modality(self):
        return self.token_grid[0] * self.token_grid[1]

    def forward(self, bev_tensors):
        tokens = []
        for modality in self.modalities:
            if modality not in bev_tensors:
                continue
            pooled = self._pool_modality_tensor(bev_tensors[modality])
            projected = self.projections[modality](pooled)
            projected = projected + self.modality_embeddings[modality]
            tokens.append(projected)
        if not tokens:
            raise ValueError("No BEV tensors were provided to the connector.")
        return torch.cat(tokens, dim=1)


class QFormerConnector(_BaseConnector):
    def __init__(
        self,
        modalities=("camera", "lidar", "fused"),
        token_grid=(4, 4),
        hidden_size=512,
        num_query_tokens=32,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_visual_tokens=True,
    ):
        super().__init__(modalities=modalities, token_grid=token_grid)
        self.hidden_size = hidden_size
        self.num_query_tokens = num_query_tokens
        self.use_visual_tokens = use_visual_tokens

        self.bev_projections = nn.ModuleDict()
        self.modality_embeddings = nn.ParameterDict()
        for modality in self.modalities:
            self.bev_projections[modality] = nn.Sequential(
                nn.LazyLinear(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
            self.modality_embeddings[modality] = nn.Parameter(
                torch.zeros(1, self.token_grid[0] * self.token_grid[1], hidden_size)
            )

        self.visual_projection = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def _encode_bev_tokens(self, bev_tensors):
        tokens = []
        for modality in self.modalities:
            if modality not in bev_tensors:
                continue
            pooled = self._pool_modality_tensor(bev_tensors[modality])
            projected = self.bev_projections[modality](pooled)
            projected = projected + self.modality_embeddings[modality]
            tokens.append(projected)
        if not tokens:
            raise ValueError("No BEV tensors were provided to the Q-Former connector.")
        return torch.cat(tokens, dim=1)

    def forward(self, bev_tensors, visual_tokens=None):
        memory_tokens = [self._encode_bev_tokens(bev_tensors)]
        if self.use_visual_tokens and visual_tokens is not None:
            if visual_tokens.dim() == 2:
                visual_tokens = visual_tokens.unsqueeze(1)
            memory_tokens.append(self.visual_projection(visual_tokens))
        memory = torch.cat(memory_tokens, dim=1)
        query = self.query_tokens.expand(memory.size(0), -1, -1)
        for layer in self.decoder_layers:
            query = layer(query, memory)
        return self.output_norm(query)
