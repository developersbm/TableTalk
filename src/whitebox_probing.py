#done by Logan Mifflin 

#This file implements a simple linear probe to predict correctness of model generations
#it uses hidden states from language models and applies platt scaling for probability calibration

# Functions:
#whiteboxprobe: neural network module for mapping hidden states to a confidence probability
#forward: takes hidden states, projects them, pools them, and returns a confidence score
#fit_platt_scaling: trains the calibration parameters a and b to make probabilities more accurate
#save: exports the basic probe parameters to a file
#save_with_pca: exports the probe along with pca components for dimensionality reduction
#load: reads a trained probe from a file back into memory
#apply_pca: reduces the dimension of hidden states if pca parameters were loaded
#train_probe_model: fully trains a whitebox probe using a dataset of hidden states and labels
#calibrate_probe: runs the platt scaling optimizer on a validation set to calibrate outputs
#extract_span_hidden_states: finds and returns only the hidden states belonging to a specific text span

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

#neural network module for mapping hidden states to a confidence probability
class WhiteBoxProbe(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        self.projection = nn.Linear(input_dim, proj_dim)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(proj_dim, 1)

        self.platt_a = nn.Parameter(torch.tensor(1.0))
        self.platt_b = nn.Parameter(torch.tensor(0.0))

        self.pca_mean: Optional[torch.Tensor] = None
        self.pca_components: Optional[torch.Tensor] = None
        self.raw_input_dim: Optional[int] = None

    #takes hidden states, projects them, pools them, and returns a confidence score
    def forward(self, hidden_states: torch.Tensor, apply_platt: bool = False) -> torch.Tensor:
        if hidden_states.size(0) == 0:
            return torch.tensor(0.5, device=hidden_states.device)
            
        proj = self.projection(hidden_states)
        
        pooled, _ = torch.max(proj, dim=0)
        
        pooled = self.dropout(pooled)
        
        logit = self.classifier(pooled)
        
        if apply_platt:
            logit = self.platt_a * logit + self.platt_b
            
        prob = torch.sigmoid(logit)
        return prob.squeeze()

    #trains the calibration parameters a and b to make probabilities more accurate
    def fit_platt_scaling(self, logits: torch.Tensor, targets: torch.Tensor, lr: float = 0.01, epochs: int = 100):
        optimizer = torch.optim.Adam([self.platt_a, self.platt_b], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            scaled_logits = self.platt_a * logits + self.platt_b
            loss = criterion(scaled_logits, targets.float())
            loss.backward()
            optimizer.step()
            
    #exports the basic probe parameters to a file
    def save(self, path: str):
        torch.save({
            'input_dim': self.input_dim,
            'proj_dim': self.proj_dim,
            'state_dict': self.state_dict(),
        }, path)

    #exports the probe along with pca components for dimensionality reduction
    def save_with_pca(
        self,
        path: str,
        pca_mean: torch.Tensor,
        pca_components: torch.Tensor,
        raw_input_dim: int,
    ):
        torch.save({
            'input_dim':       self.input_dim,
            'proj_dim':        self.proj_dim,
            'state_dict':      self.state_dict(),
            'pca_mean':        pca_mean.cpu(),
            'pca_components':  pca_components.cpu(),
            'raw_input_dim':   raw_input_dim,
        }, path)

    #reads a trained probe from a file back into memory
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'WhiteBoxProbe':
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        probe = cls(checkpoint['input_dim'], checkpoint['proj_dim'])
        probe.load_state_dict(checkpoint['state_dict'])
        probe.to(device)
        if 'pca_mean' in checkpoint and checkpoint['pca_mean'] is not None:
            probe.pca_mean = checkpoint['pca_mean'].to(device)
            probe.pca_components = checkpoint['pca_components'].to(device)
            probe.raw_input_dim = int(checkpoint.get('raw_input_dim', probe.pca_mean.shape[0] if probe.pca_mean is not None else checkpoint['input_dim']))
        return probe

    #reduces the dimension of hidden states if pca parameters were loaded
    def apply_pca(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pca_mean is None or self.pca_components is None:
            return hidden_states
        hs = hidden_states.to(self.pca_mean.device).float()
        centered = hs - self.pca_mean.unsqueeze(0)
        reduced = centered @ self.pca_components.T
        return reduced

#fully trains a whitebox probe using a dataset of hidden states and labels
def train_probe_model(
    train_data: List[Tuple[torch.Tensor, int]], 
    input_dim: int,
    proj_dim: int = 64,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32
) -> WhiteBoxProbe:
    probe = WhiteBoxProbe(input_dim=input_dim, proj_dim=proj_dim)
    optimizer = torch.optim.Adam([p for n, p in probe.named_parameters() if 'platt' not in n], lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    import random
    random.shuffle(train_data)
    
    probe.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, (hs, label) in enumerate(train_data):
            if hs.size(0) == 0:
                continue
            
            proj = probe.projection(hs)
            pooled, _ = torch.max(proj, dim=0)
            pooled = probe.dropout(pooled)
            logit = probe.classifier(pooled)
            
            prob = torch.sigmoid(logit)
            loss = criterion(prob.unsqueeze(0), torch.tensor([[float(label)]]))
            
            epoch_loss += loss.item()
            
            (loss / batch_size).backward()
            
            if (i + 1) % batch_size == 0 or (i + 1) == len(train_data):
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"Epoch {ep+1}/{epochs} Loss: {epoch_loss / len(train_data):.4f}")
        
    return probe

#runs the platt scaling optimizer on a validation set to calibrate outputs
def calibrate_probe(probe: WhiteBoxProbe, val_data: List[Tuple[torch.Tensor, int]]):
    probe.eval()
    with torch.no_grad():
        logits = []
        targets = []
        for hs, label in val_data:
            if hs.size(0) == 0:
                continue
            proj = probe.projection(hs)
            pooled, _ = torch.max(proj, dim=0)
            logit = probe.classifier(pooled)
            logits.append(logit.squeeze())
            targets.append(label)
            
    if logits:
        logits_t = torch.stack(logits)
        targets_t = torch.tensor(targets)
        probe.fit_platt_scaling(logits_t, targets_t)

#finds and returns only the hidden states belonging to a specific text span
def extract_span_hidden_states(
    text: str,
    tokens: List[str],
    hidden_states: torch.Tensor,
    span: str
) -> torch.Tensor:
    if hidden_states is None or hidden_states.size(0) == 0:
        return torch.empty((0,))
        
    
    char_idx = 0
    token_indices = []
    
    span_start = text.find(span)
    if span_start == -1:
        return hidden_states
        
    span_end = span_start + len(span)
    
    current_char_pos = 0
    for i, tok in enumerate(tokens):
        tok_len = len(tok)
        tok_start = current_char_pos
        tok_end = current_char_pos + tok_len
        
        if max(tok_start, span_start) < min(tok_end, span_end):
            token_indices.append(i)
            
        current_char_pos += tok_len
        
    if not token_indices:
        return torch.empty((0,))
        
    span_hs = hidden_states[token_indices]
    return span_hs
