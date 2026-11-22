from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


class TSMixerBlock(nn.Module):
    def __init__(self, time_steps: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_norm = nn.LayerNorm(feature_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_steps, time_steps),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_steps, time_steps),
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.time_norm(x)
        y = y.transpose(1, 2)
        y = self.time_mlp(y).transpose(1, 2)
        x = residual + y
        x = x + self.feature_mlp(self.feature_norm(x))
        return x


class IntradayPatchBranch(nn.Module):
    def __init__(
        self,
        history: int,
        daily_feat_dim: int,
        hidden_dim: int = 64,
        patch_sizes: List[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_sizes = patch_sizes or [4, 8, 16]
        self.patch_projs = nn.ModuleList([nn.Linear(96 // patch, hidden_dim) for patch in self.patch_sizes])
        self.curve_proj = nn.Linear(96, hidden_dim)
        self.daily_proj = nn.Linear(daily_feat_dim, hidden_dim)
        self.scale_gate = nn.Sequential(
            nn.Linear(hidden_dim * (2 + len(self.patch_sizes)), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.patch_sizes) + 2),
        )
        self.mixer = nn.Sequential(
            TSMixerBlock(history, hidden_dim, dropout),
            TSMixerBlock(history, hidden_dim, dropout),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

    def forward(self, curve_hist: torch.Tensor, daily_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, steps, _ = curve_hist.shape
        curve_flat = curve_hist.reshape(bsz * steps, 96)
        scale_tokens = []
        for patch, proj in zip(self.patch_sizes, self.patch_projs):
            pooled = curve_flat.reshape(bsz * steps, 96 // patch, patch).mean(dim=-1)
            scale_tokens.append(proj(pooled).reshape(bsz, steps, -1))
        raw_token = self.curve_proj(curve_hist)
        daily_token = self.daily_proj(daily_hist)
        token_stack = torch.cat([raw_token, daily_token, *scale_tokens], dim=-1)
        gate_logits = self.scale_gate(token_stack)
        gate = torch.softmax(gate_logits, dim=-1)
        components = [raw_token, daily_token, *scale_tokens]
        mixed = sum(gate[..., i : i + 1] * comp for i, comp in enumerate(components))
        tokens = self.mixer(mixed)
        context = self.context_norm(tokens[:, -1] + tokens.mean(dim=1))
        return context, tokens


class DynamicIndustryGraphBranch(nn.Module):
    def __init__(
        self,
        history: int,
        num_nodes: int,
        global_dim: int,
        hidden_dim: int = 64,
        topk: int = 8,
        use_dynamic_graph: bool = True,
        use_static_adjacency: bool = True,
        learnable_static_adjacency: bool = True,
        use_topk: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.history = history
        self.topk = min(topk, num_nodes)
        self.use_dynamic_graph = use_dynamic_graph
        self.use_static_adjacency = use_static_adjacency
        self.use_topk = use_topk
        self.node_proj = nn.Sequential(
            nn.Linear(history, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.static_adj = nn.Parameter(
            torch.zeros(num_nodes, num_nodes),
            requires_grad=learnable_static_adjacency,
        )
        self.graph_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.global_encoder = nn.GRU(global_dim, hidden_dim, batch_first=True)
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _dynamic_adjacency(self, node_embed: torch.Tensor) -> torch.Tensor:
        if self.use_dynamic_graph:
            normed = torch.nn.functional.normalize(node_embed, dim=-1)
            adj = torch.matmul(normed, normed.transpose(1, 2))
        else:
            base = self.static_adj
            adj = base.unsqueeze(0).expand(node_embed.size(0), -1, -1)
        if self.use_static_adjacency and self.use_dynamic_graph:
            base = self.static_adj
            adj = adj + base.unsqueeze(0)
        if self.use_topk and self.topk < self.num_nodes:
            topk = torch.topk(adj, k=self.topk, dim=-1).indices
            mask = torch.full_like(adj, float("-inf"))
            mask.scatter_(-1, topk, adj.gather(-1, topk))
            adj = mask
        return torch.softmax(adj, dim=-1)

    def forward(self, industry_hist: torch.Tensor, global_hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_hist = industry_hist.transpose(1, 2)
        node_embed = self.node_proj(node_hist)
        adj = self._dynamic_adjacency(node_embed)
        h = torch.relu(self.graph_linear1(torch.matmul(adj, node_embed)))
        h = torch.relu(self.graph_linear2(torch.matmul(adj, h)))
        _, hidden = self.global_encoder(global_hist)
        global_ctx = hidden[-1]
        gate_input = torch.cat([h, global_ctx.unsqueeze(1).expand(-1, h.size(1), -1)], dim=-1)
        weights = torch.softmax(self.readout_gate(gate_input), dim=1)
        graph_ctx = (weights * h).sum(dim=1)
        context = self.output_norm(graph_ctx + global_ctx)
        return context, h, adj, weights.squeeze(-1)


class CrossGuidedFusion(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.day_query_month = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.month_query_day = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        day_context: torch.Tensor,
        day_tokens: torch.Tensor,
        month_context: torch.Tensor,
        month_tokens: torch.Tensor,
    ) -> torch.Tensor:
        day_query = day_context.unsqueeze(0)
        month_query = month_context.unsqueeze(0)
        month_tokens_t = month_tokens.transpose(0, 1)
        day_tokens_t = day_tokens.transpose(0, 1)
        day_guided, _ = self.day_query_month(day_query, month_tokens_t, month_tokens_t)
        month_guided, _ = self.month_query_day(month_query, day_tokens_t, day_tokens_t)
        concat = torch.cat(
            [day_context, month_context, day_guided.squeeze(0), month_guided.squeeze(0)],
            dim=-1,
        )
        return self.norm(self.fusion(concat))


class ReconciliationHead(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim * 3 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(global_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, global_hist: torch.Tensor) -> torch.Tensor:
        last_day = global_hist[:, -1]
        recent_mean = global_hist[:, -7:].mean(dim=1)
        overall_mean = global_hist.mean(dim=1)
        sales_gap = global_hist[:, -1, 7:8] if global_hist.size(-1) > 8 else global_hist[:, -1, 2:3]
        scalar_stats = torch.cat(
            [
                global_hist[:, -1:, 0].mean(dim=1, keepdim=True),
                global_hist[:, -1:, 5 if global_hist.size(-1) > 8 else 1].mean(dim=1, keepdim=True),
                global_hist[:, -1:, 7 if global_hist.size(-1) > 8 else 2].mean(dim=1, keepdim=True),
                global_hist[:, -1:, 8 if global_hist.size(-1) > 8 else 3].mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        features = torch.cat([last_day, recent_mean, overall_mean, scalar_stats], dim=-1)
        residual = self.net(features)
        gate = self.gate(torch.cat([last_day, sales_gap], dim=-1))
        return gate * residual


class StatisticalAnchorHead(nn.Module):
    def __init__(self, global_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.sales_ratio_head = nn.Sequential(
            nn.Linear(global_dim * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.blend_gate = nn.Sequential(
            nn.Linear(global_dim * 2 + 5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, global_hist: torch.Tensor, predicted_industry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last_day = global_hist[:, -1]
        mean3 = global_hist[:, -3:].mean(dim=1)
        mean7 = global_hist[:, -7:].mean(dim=1)
        load_prev1 = global_hist[:, -1, 0:1]
        load_prev7 = global_hist[:, -7, 0:1] if global_hist.size(1) >= 7 else load_prev1
        sales_col = 5 if global_hist.size(-1) > 8 else 1
        sales_prev1 = global_hist[:, -1, sales_col : sales_col + 1]
        peak_prev1 = global_hist[:, -1, 1:2] if global_hist.size(-1) > 8 else sales_prev1
        mean96_prev1 = global_hist[:, -1, 3:4] if global_hist.size(-1) > 8 else sales_prev1
        stats = torch.cat(
            [last_day, mean3, mean7, load_prev1, load_prev7, peak_prev1, mean96_prev1, sales_prev1, predicted_industry[:, :1]],
            dim=-1,
        )
        pred_ratio = self.sales_ratio_head(stats)
        anchor_from_industry = predicted_industry[:, :1] * (1.0 + 0.02 * pred_ratio)
        blend_logits = self.blend_gate(
            torch.cat([last_day, mean7, load_prev1, load_prev7, peak_prev1, mean96_prev1, predicted_industry[:, :1]], dim=-1)
        )
        blend = torch.softmax(blend_logits, dim=-1)
        persistence_anchor = blend[:, :1] * load_prev1 + blend[:, 1:2] * load_prev7
        anchor = 0.7 * anchor_from_industry + 0.3 * persistence_anchor
        return anchor, pred_ratio


class TwoStageDeltaHead(nn.Module):
    def __init__(self, global_dim: int, industry_nodes: int, calendar_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.delta_head = nn.Sequential(
            nn.Linear(global_dim * 4 + industry_nodes + calendar_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(global_dim * 2 + industry_nodes + calendar_dim + 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        global_hist: torch.Tensor,
        predicted_industry: torch.Tensor,
        target_calendar: torch.Tensor,
        branch_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_day = global_hist[:, -1]
        prev_day = global_hist[:, -2] if global_hist.size(1) >= 2 else last_day
        mean3 = global_hist[:, -3:].mean(dim=1)
        mean7 = global_hist[:, -7:].mean(dim=1)
        load_delta1 = last_day[:, 0:1] - prev_day[:, 0:1]
        load_delta7 = last_day[:, 0:1] - (global_hist[:, -7, 0:1] if global_hist.size(1) >= 7 else prev_day[:, 0:1])
        sales_col = 5 if global_hist.size(-1) > 8 else 1
        sales_delta = predicted_industry[:, :1] - last_day[:, sales_col : sales_col + 1]
        features = torch.cat(
            [
                last_day,
                mean3,
                mean7,
                last_day - prev_day,
                predicted_industry,
                target_calendar,
                branch_delta,
                load_delta1,
                load_delta7,
                sales_delta,
            ],
            dim=-1,
        )
        delta = self.delta_head(features)
        gate = self.gate(torch.cat([last_day, mean7, predicted_industry, target_calendar, branch_delta, sales_delta], dim=-1))
        return delta, gate


@dataclass
class ModelConfig:
    intraday_history: int = 14
    industry_history: int = 28
    intraday_feat_dim: int = 15
    industry_nodes: int = 19
    global_dim: int = 14
    target_calendar_dim: int = 7
    hidden_dim: int = 64
    patch_sizes: List[int] | None = None
    dropout: float = 0.1
    num_heads: int = 4
    use_intraday_branch: bool = True
    use_industry_graph_branch: bool = True
    use_dynamic_graph: bool = True
    use_static_adjacency: bool = True
    learnable_static_adjacency: bool = True
    use_topk: bool = True
    use_cross_guided_fusion: bool = True
    use_statistical_anchor: bool = True
    use_reconciliation: bool = True


class DualScaleLoadForecastModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.intraday_branch = IntradayPatchBranch(
            history=config.intraday_history,
            daily_feat_dim=config.intraday_feat_dim,
            hidden_dim=config.hidden_dim,
            patch_sizes=config.patch_sizes or [4, 8, 16],
            dropout=config.dropout,
        )
        self.industry_branch = DynamicIndustryGraphBranch(
            history=config.industry_history,
            num_nodes=config.industry_nodes,
            global_dim=config.global_dim,
            hidden_dim=config.hidden_dim,
            topk=min(8, config.industry_nodes),
            use_dynamic_graph=config.use_dynamic_graph,
            use_static_adjacency=config.use_static_adjacency,
            learnable_static_adjacency=config.learnable_static_adjacency,
            use_topk=config.use_topk,
        )
        self.fusion = CrossGuidedFusion(config.hidden_dim, config.num_heads, config.dropout)
        self.target_calendar_proj = nn.Linear(config.target_calendar_dim, config.hidden_dim)
        self.flat_proj = nn.Linear(96 + config.intraday_feat_dim + config.industry_nodes + config.global_dim, config.hidden_dim)
        self.flat_backbone = nn.Sequential(
            TSMixerBlock(config.industry_history, config.hidden_dim, config.dropout),
            TSMixerBlock(config.industry_history, config.hidden_dim, config.dropout),
        )
        self.industry_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + config.target_calendar_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.industry_nodes),
        )
        self.anchor_head = StatisticalAnchorHead(config.global_dim, config.hidden_dim)
        self.two_stage_head = TwoStageDeltaHead(
            config.global_dim,
            config.industry_nodes,
            config.target_calendar_dim,
            config.hidden_dim,
        )
        self.main_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 3 + config.industry_nodes + 1, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 4),
        )
        self.reconciliation = ReconciliationHead(config.global_dim, config.hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        bsz = batch["target_calendar"].size(0)
        device = batch["target_calendar"].device
        if self.config.use_intraday_branch:
            day_context, day_tokens = self.intraday_branch(batch["intraday_curve_hist"], batch["intraday_daily_hist"])
        else:
            day_context = torch.zeros(bsz, self.config.hidden_dim, device=device)
            day_tokens = torch.zeros(bsz, self.config.intraday_history, self.config.hidden_dim, device=device)
        if self.config.use_industry_graph_branch:
            industry_context, industry_tokens, dynamic_adj, node_importance = self.industry_branch(batch["industry_hist"], batch["global_hist"])
        else:
            industry_context = torch.zeros(bsz, self.config.hidden_dim, device=device)
            industry_tokens = torch.zeros(bsz, self.config.industry_nodes, self.config.hidden_dim, device=device)
            dynamic_adj = torch.zeros(bsz, self.config.industry_nodes, self.config.industry_nodes, device=device)
            node_importance = torch.zeros(bsz, self.config.industry_nodes, device=device)
        if self.config.use_cross_guided_fusion:
            fused = self.fusion(day_context, day_tokens, industry_context, industry_tokens)
        else:
            raise RuntimeError("Original-checkpoint reconstruction supports the original full cross-guided fusion only.")
        target_ctx = self.target_calendar_proj(batch["target_calendar"])
        flat_tokens = self.flat_backbone(self.flat_proj(batch["flat_hist"]))
        flat_context = flat_tokens[:, -1] + flat_tokens.mean(dim=1)
        industry_prediction = self.industry_decoder(torch.cat([industry_context, flat_context, batch["target_calendar"]], dim=-1))
        if self.config.use_statistical_anchor:
            anchor, pred_ratio = self.anchor_head(batch["global_hist"], industry_prediction)
        else:
            anchor = torch.zeros(bsz, 1, device=device)
            pred_ratio = torch.zeros(bsz, 1, device=device)
        branch_prediction = self.main_head(torch.cat([fused, target_ctx, flat_context, industry_prediction, pred_ratio], dim=-1))
        if self.config.use_reconciliation:
            residual = self.reconciliation(batch["global_hist"])
        else:
            residual = torch.zeros_like(anchor)
        main_load = anchor + branch_prediction[:, :1] + residual
        prediction = torch.cat([main_load, branch_prediction[:, 1:]], dim=-1)
        return {
            "prediction": prediction,
            "industry_prediction": industry_prediction,
            "dynamic_adj": dynamic_adj,
            "node_importance": node_importance,
            "fused_context": fused,
        }
