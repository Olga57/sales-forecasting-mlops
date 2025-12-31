import torch
import torch.nn as nn
import torch.nn.functional as func


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def smape(y_true, y_pred):
    return 100 * torch.mean(
        2
        * torch.abs(y_pred - y_true)
        / (torch.abs(y_pred) + torch.abs(y_true) + 1e-8)
    )


def wape(y_true, y_pred):
    return torch.sum(torch.abs(y_true - y_pred)) / torch.sum(torch.abs(y_true))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = x
        x = func.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        gated = torch.sigmoid(self.gate(x)) * x
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, hidden_size):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size
        self.grns = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_size) for _ in range(num_vars)]
        )
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        var_outputs = []
        for i in range(self.num_vars):
            var_input = x[:, :, i, :]
            var_out = self.grns[i](var_input)
            var_outputs.append(var_out.unsqueeze(2))

        var_outputs = torch.cat(var_outputs, dim=2)

        batch_size, seq_len, num_vars, hidden_size = var_outputs.shape

        flat = var_outputs.view(-1, hidden_size)
        attn_scores = self.attn(flat)
        attn_scores = attn_scores.view(batch_size, seq_len, num_vars, 1)
        attn_weights = torch.softmax(attn_scores, dim=2)

        vsn_out = (var_outputs * attn_weights).sum(dim=2)
        return vsn_out, attn_weights


class TFTModel(nn.Module):
    def __init__(self, cfg, num_vars: int):
        super().__init__()

        self.vsn = VariableSelectionNetwork(
            num_vars=num_vars, hidden_size=cfg.hidden_size
        )

        self.encoder_lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            batch_first=True,
        )

        self.decoder_lstm = nn.LSTM(
            input_size=cfg.hidden_size,
            hidden_size=cfg.hidden_size,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.num_heads,
            batch_first=True,
        )

        self.grn = GatedResidualNetwork(
            cfg.hidden_size, cfg.hidden_size, dropout=cfg.dropout
        )
        self.pred_head = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x):
        vsn_out, attn_weights = self.vsn(x)
        enc_out, _ = self.encoder_lstm(vsn_out)
        dec_out, _ = self.decoder_lstm(enc_out)
        attn_out, _ = self.attention(dec_out, dec_out, dec_out)
        grn_out = self.grn(attn_out)
        out = self.pred_head(grn_out[:, -1, :])
        return out, attn_weights
