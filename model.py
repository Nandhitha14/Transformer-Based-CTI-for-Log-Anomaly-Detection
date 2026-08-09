import torch
import torch.nn as nn

class TransformerTCN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len=1, dim)
        trans_out = self.transformer(x.permute(1,0,2))[-1]  # (batch, dim)
        tcn_out = self.tcn(x.permute(0,2,1)).mean(dim=2)    # (batch, hidden_dim)
        fused = torch.cat([trans_out, tcn_out], dim=1)
        return self.fc(fused)
