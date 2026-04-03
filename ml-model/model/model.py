
import numpy as np
import torch
import torch.nn as nn



def strokes_to_5d(strokes):
    seq = []
    
    for stroke in strokes:
        x, y = stroke
        
        for i in range(len(x)):
            if i == 0:
                dx, dy = 0, 0
            else:
                dx = (x[i] - x[i-1]) / 255.0
                dy = (y[i] - y[i-1]) / 255.0
            
            seq.append([dx, dy, 1, 0, 0])
        
        seq[-1][2] = 0
        seq[-1][3] = 1
    
    seq.append([0, 0, 0, 0, 1])
    return seq

def pad_sequence(seq, max_len=200):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [[0,0,0,0,0]] * (max_len - len(seq))

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

model = DoodleModel(num_classes=200)
model.load_state_dict(torch.load("rnn_model.pth", map_location="cpu"))
model.eval()

MAX_SEQ_LEN = 200  # same as training

def preprocess(strokes):
    seq = strokes_to_5d(strokes)
    length = min(len(seq), MAX_SEQ_LEN)
    seq = pad_sequence(seq, MAX_SEQ_LEN)

    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    length = torch.tensor([length])

    return seq, length

def predict(strokes):
    seq, length = preprocess(strokes)

    with torch.no_grad():
        output, _ = model(seq, length)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred]

def predict_topk(strokes, k=3):
    seq, length = preprocess(strokes)

    with torch.no_grad():
        output, _ = model(seq, length)
        probs = torch.softmax(output, dim=1)
        topk = torch.topk(probs, k)

    return [
        {"class": class_names[i], "prob": float(p)}
        for i, p in zip(topk.indices[0], topk.values[0])
    ]
