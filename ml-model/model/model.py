import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_out, lengths):
        
        B, T, _ = lstm_out.size()
        
        scores = torch.tanh(self.attn(lstm_out))
        scores = self.context(scores).squeeze(-1)  # (B, T)
        
        
        device = lengths.device
        mask = torch.arange(T, device=device).expand(B, T) < lengths.unsqueeze(1)
        
        
        scores = scores.masked_fill(~mask, -1e4)
        
        weights = torch.softmax(scores, dim=1)
        
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        
        return context, weights


class DoodleModel(nn.Module):

    def __init__(self, input_size=5, hidden_size=256, num_classes=200):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True
        )

        self.attn = Attention(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self,x,lengths,hidden=None):

        lstm_out,hidden = self.lstm(x,hidden)

        idx = (lengths-1).unsqueeze(1).unsqueeze(2).expand(-1,1,lstm_out.size(2))
        last_hidden = lstm_out.gather(1,idx).squeeze(1)

        context,weights = self.attn(lstm_out,lengths)

        combined = context + last_hidden

        out = self.fc(combined)

        return out,weights