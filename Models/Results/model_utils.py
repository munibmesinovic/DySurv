import json
import matplotlib.pyplot as plt

from src.data_processing.data_loader import data_loader
import src.models.model_utils as model_utils
from src.results.main import evaluate
import src.visualisation.main as vis_main

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import numba
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from src.models.survival.metrics import concordance_td

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
from torch import nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

def get_exp_dir(data_load_config=None, model_name : str = "DySurv"):
    
    # Standardise model_name
    if "dysurv" in model_name.lower():
        model_name = "DySurv"
    elif "ddh" in model_name.lower():
        model_name = "DynamicDeephit"
    elif "deephit" in model_name.lower():
        model_name = "Deephit"
    else:
        model_name = "Test"

    # Get save path
    data_name = data_load_config["data_name"]
    run_num = 1
    SAVE_FD = f"experiments/{data_name}/{model_name}/results_{data_load_config['seed']}/"

    if not os.path.exists(SAVE_FD):
        os.makedirs(SAVE_FD)
    while os.path.exists(SAVE_FD + f"run{run_num}/"):
        run_num += 1

    save_fd = SAVE_FD + f"run{run_num}/"
    if not os.path.exists(save_fd):
        os.makedirs(save_fd)

    return save_fd

class extract_tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        if type(x) is tuple:
            tensor, _ = x
        else:
            tensor = x
        # Reshape shape (batch, hidden)
        if tensor.dim() == 2:
            return tensor[:, :]
        return tensor.mean(dim=1)
        #return tensor[:, -1, :] # original, now instead of returning just the last bit, I return the mean of the values.
    

class Decoder(nn.Module):
    def __init__(self, seq_len, no_features, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.hidden_size,
            num_layers = 1,
            batch_first = True
        )
        self.dropout = nn.Dropout()

        self.fc1 = nn.Linear(self.hidden_size, 3*self.hidden_size)
        self.fc2 = nn.Linear(3*self.hidden_size, 5*self.hidden_size)
        self.fc3 = nn.Linear(5*self.hidden_size, 3*self.hidden_size)
        self.fc4 = nn.Linear(3*self.hidden_size, output_size)
        
    def forward(self, x, y):
        x = torch.cat((x, y.reshape(-1, 1)), dim=1)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        out = self.fc4(x)
        return out
    
class DySurv(nn.Module):
    def __init__(self, in_features, encoded_features, out_features, seq_len):
        super().__init__()
        
        self.lstm1 = nn.LSTM(in_features, in_features, batch_first=True)
        self.extract = extract_tensor() # self.extract is there for 
        self.fc11 = nn.Linear(in_features, 3*in_features)
        self.fc12 = nn.Linear(3*in_features, 5*in_features)
        self.fc13 = nn.Linear(5*in_features, 3*in_features)
        self.fc14 = nn.Linear(3*in_features, encoded_features)

        self.fc24 = nn.Linear(3*in_features, encoded_features)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout()

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 3*in_features), nn.ReLU(), 
            nn.Linear(3*in_features, 5*in_features), nn.ReLU(), 
            nn.Linear(5*in_features, 3*in_features), nn.ReLU(), 
            nn.Linear(3*in_features, out_features),
        )
        
        # Adapt number of units
        self.decoder2 = Decoder(seq_len, encoded_features+1, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        sample_z = eps.mul(std).add_(mu)

        return sample_z
    
    def encoder(self, x):
        # Variation - add padding
        padding_value = -0.1
        mask = torch.isclose(x, torch.tensor(padding_value, dtype=x.dtype), atol=1e-4)
        mask = mask.all(dim=-1)
        lengths = (~mask).sum(dim=1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM        
        x, _ = self.lstm1(x)

        # Unpack
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.relu(self.fc11(self.extract(x))) 
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z
    
    def forward(self, x):
        # Needs to be adapted
        y = x[:, -1, -1]
        x = x[:, :, :-1] # stuff that goes through the encoder
        
        # Encoder pass
        mu, logvar = self.encoder(x.float())

        # Reparametrisation
        z = self.reparameterize(mu, logvar)
        
        # Decoder pass
        # We pass y back, to see whether the event is predicted too
        return self.decoder2(z, y.float()), self.surv_net(z), mu, logvar

    def predict(self, input):
        mu, logvar = self.encoder(input)
        encoded = self.reparameterize(mu, logvar)

        return self.surv_net(encoded)


class DySurvCompetingRisk(nn.Module):
    """
    DySurv model extended to handle competing risks.

    This variant outputs separate hazard predictions for each competing event type.
    The architecture is similar to the original DySurv but with multiple output heads,
    one for each risk.

    Args:
        in_features: Number of input features
        encoded_features: Dimension of latent space
        out_features: Number of time points for discretization
        seq_len: Length of input sequences
        num_risks: Number of competing event types (K)
    """
    def __init__(self, in_features, encoded_features, out_features, seq_len, num_risks=2):
        super().__init__()

        self.num_risks = num_risks

        # Encoder components (shared across all risks)
        self.lstm1 = nn.LSTM(in_features, in_features, batch_first=True)
        self.extract = extract_tensor()
        self.fc11 = nn.Linear(in_features, 3*in_features)
        self.fc12 = nn.Linear(3*in_features, 5*in_features)
        self.fc13 = nn.Linear(5*in_features, 3*in_features)
        self.fc14 = nn.Linear(3*in_features, encoded_features)
        self.fc24 = nn.Linear(3*in_features, encoded_features)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        # Multiple survival networks - one for each competing risk
        # Each outputs hazard predictions for all time points
        self.surv_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoded_features, 3*in_features), nn.ReLU(),
                nn.Linear(3*in_features, 5*in_features), nn.ReLU(),
                nn.Linear(5*in_features, 3*in_features), nn.ReLU(),
                nn.Linear(3*in_features, out_features),
            )
            for _ in range(num_risks)
        ])

        # Decoder (same as original)
        self.decoder2 = Decoder(seq_len, encoded_features+1, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        sample_z = eps.mul(std).add_(mu)
        return sample_z

    def encoder(self, x):
        # Variation - add padding
        padding_value = -0.1
        mask = torch.isclose(x, torch.tensor(padding_value, dtype=x.dtype), atol=1e-4)
        mask = mask.all(dim=-1)
        lengths = (~mask).sum(dim=1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        x, _ = self.lstm1(x)

        # Unpack
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.relu(self.fc11(self.extract(x)))
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z

    def forward(self, x):
        # Extract event indicator (last feature)
        y = x[:, -1, -1]
        x = x[:, :, :-1]  # Input to encoder

        # Encoder pass
        mu, logvar = self.encoder(x.float())

        # Reparametrization
        z = self.reparameterize(mu, logvar)

        # Multiple survival predictions - one per risk
        # Stack outputs: [batch, time_points, num_risks]
        surv_outputs = torch.stack([surv_net(z) for surv_net in self.surv_nets], dim=2)

        # Decoder pass
        decoded = self.decoder2(z, y.float())

        return decoded, surv_outputs, mu, logvar

    def predict(self, input):
        """
        Predict competing risk hazards.

        Returns:
            Tensor of shape [batch, time_points, num_risks]
        """
        mu, logvar = self.encoder(input)
        encoded = self.reparameterize(mu, logvar)

        # Stack predictions from all risk-specific networks
        surv_outputs = torch.stack([surv_net(encoded) for surv_net in self.surv_nets], dim=2)
        return surv_outputs


######## Loss ########

def adaptive_pos_weight(events) -> float:
    """Compute pos_weight from event rate, auto-adapting to dataset imbalance.

    Inverse of event rate, capped at 20. For high event-rate datasets (>=30%)
    no reweighting is applied. This replaces the old hardcoded pos_weight=50.
    """
    if isinstance(events, Tensor):
        event_rate = (events > 0).float().mean().item()
    else:
        event_rate = float((np.array(events) > 0).mean())
    if event_rate < 0.3:
        return min(max(1.0 / (event_rate + 1e-3), 1.0), 20.0)
    return 1.0


def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean', training: bool = True,
                        pos_weight: float | None = None) -> Tensor:
    """
    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf

    Inputs:
        phi - network output
        idx_durations - tensor, recording the time at which an event or censoring occured. [time_1, time_2, etc]
        events - tensor, indicating whether the event occured (1) or censoring happened (0). [1, 1, 0, 1, 0, etc]
    Output:
        loss - scalar, reducted tensor, of the BCE loss along the time-axis for each patient
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    
    # Change type of events if necessary
    if events.dtype is torch.bool:
        events = events.float()
    
    # Change views
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    
    # Creates a target for bce: initialise everything with 0, and setting events at idx_duration
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    
    # Adaptive pos_weight: auto-set from event rate instead of hardcoded value
    pw = torch.tensor([pos_weight if pos_weight is not None else 1.0], device=phi.device)

    # Compute BCE
    if training:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, pos_weight=pw, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')

    # Compute the loss, along the time axis, ie for each patient separately
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    
    # Take the mean or something of the loss in the end
    return _reduction(loss, reduction)

class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

class NLLLogistiHazardLoss(_Loss):
    def __init__(self, reduction: str = 'mean', pos_weight: float | None = None) -> None:
        super().__init__(reduction)
        self.pos_weight = pos_weight

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction, self.training,
                                   pos_weight=self.pos_weight)
    

######## Competing Risk Loss Functions ########

def nll_competing_risk_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                               num_risks: int, reduction: str = 'mean',
                               training: bool = True) -> Tensor:
    """
    Negative log-likelihood for competing risks using cause-specific hazards.

    This loss function models K competing events using separate hazard functions.
    Each event type has its own hazard, and the loss properly accounts for:
    - The hazard of the specific event that occurred
    - The survival from all other competing events
    - Censoring (when event = 0)

    Args:
        phi: Tensor of shape [batch_size, num_time_points, num_risks]
             Network output with logits for each risk's hazard at each time point
        idx_durations: Tensor of shape [batch_size]
                      Time index at which event or censoring occurred
        events: Tensor of shape [batch_size]
               Event type indicator (0=censored, 1=event_1, 2=event_2, ..., K=event_K)
        num_risks: Integer, number of competing risk types (K)
        reduction: str, 'mean', 'sum', or 'none'
        training: bool, whether in training mode (affects weighting)

    Returns:
        loss: Scalar tensor (if reduction='mean' or 'sum') or tensor of shape [batch_size]

    Mathematical formulation:
        For individual i experiencing event k at time t_i:
            L_i = log(h_k(t_i)) + sum_{j=1}^{t_i} sum_{r=1}^{K} log(1 - h_r(j))

        For individual i censored at time t_i:
            L_i = sum_{j=1}^{t_i} sum_{r=1}^{K} log(1 - h_r(j))

    Reference:
        Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018).
        DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks.
        AAAI Conference on Artificial Intelligence.
    """
    batch_size = phi.shape[0]
    num_time_points = phi.shape[1]

    # Validate dimensions
    if phi.shape[2] != num_risks:
        raise ValueError(f"Network output `phi` has {phi.shape[2]} risk channels, "
                        f"but `num_risks` = {num_risks}")

    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` has {phi.shape[1]} time points, "
                        f"but max `idx_durations` = {idx_durations.max().item()}")

    # Ensure correct dtypes
    if events.dtype is torch.bool:
        events = events.long()
    elif events.dtype is torch.float:
        events = events.long()

    # Reshape for broadcasting
    idx_durations = idx_durations.view(-1, 1, 1)  # [batch, 1, 1]
    events = events.view(-1)  # [batch]

    # Convert logits to probabilities: h(t) = sigmoid(phi(t))
    hazards = torch.sigmoid(phi)  # [batch, time, risks]

    # Compute survival probabilities: S(t) = prod(1 - h(j)) for j <= t
    # log(S(t)) = sum(log(1 - h(j))) for j <= t
    log_survival = torch.log(1 - hazards + 1e-7)  # [batch, time, risks] - add epsilon for numerical stability

    # Create time mask: 1 for times <= event time, 0 otherwise
    time_range = torch.arange(num_time_points, device=phi.device).view(1, -1, 1)  # [1, time, 1]
    time_mask = (time_range <= idx_durations).float()  # [batch, time, 1]

    # Sum log survival probabilities up to event time for all risks
    # This gives: sum_{j=1}^{t_i} sum_{r=1}^{K} log(1 - h_r(j))
    cumulative_log_survival = (log_survival * time_mask).sum(dim=1)  # [batch, risks]
    total_log_survival = cumulative_log_survival.sum(dim=1)  # [batch]

    # For individuals who experienced an event (not censored)
    # Add the log hazard of the specific event type that occurred
    # log(h_k(t_i)) where k is the event type
    event_occurred = events > 0  # [batch]

    # Get log hazard for the specific event type at the event time
    # First, gather the hazards at the event time for all risks
    idx_durations_squeezed = idx_durations.squeeze(-1)  # [batch, 1]
    event_time_hazards = hazards.gather(1, idx_durations_squeezed.expand(-1, num_risks).unsqueeze(1)).squeeze(1)  # [batch, risks]

    # Create one-hot encoding for event types (excluding censoring = 0)
    # event=1 -> [1,0,0,...], event=2 -> [0,1,0,...], etc.
    event_mask = torch.zeros((batch_size, num_risks), device=phi.device)
    if event_occurred.any():
        valid_events = events[event_occurred] - 1  # Convert to 0-indexed (event 1 -> index 0)
        event_mask[event_occurred] = F.one_hot(valid_events, num_classes=num_risks).float()

    # Log hazard for the specific event type
    log_event_hazard = torch.log(event_time_hazards + 1e-7)  # [batch, risks]
    log_event_hazard_specific = (log_event_hazard * event_mask).sum(dim=1)  # [batch]

    # Complete loss computation
    # For censored: L = sum log(1 - h(t)) for all t <= censoring time
    # For event k: L = log(h_k(t)) + sum log(1 - h(t)) for all t <= event time
    loss = -(total_log_survival + log_event_hazard_specific * event_occurred.float())

    # Optional: Add class balancing weights during training
    if training and num_risks > 1:
        # Weight events more heavily if specified
        weights = torch.ones_like(loss)
        weights[event_occurred] = 2.0  # Give more weight to actual events vs censoring
        loss = loss * weights

    return _reduction(loss, reduction)


class NLLCompetingRiskLoss(_Loss):
    """
    Competing risk loss module wrapper.

    This loss handles multiple competing events using cause-specific hazards.

    Args:
        num_risks: Number of competing event types (K)
        reduction: How to reduce the loss ('mean', 'sum', or 'none')
    """
    def __init__(self, num_risks: int, reduction: str = 'mean'):
        super().__init__(reduction)
        self.num_risks = num_risks

    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        """
        Forward pass for competing risk loss.

        Args:
            phi: [batch, time_points, num_risks] - predicted hazards for each risk
            idx_durations: [batch] - time indices of events/censoring
            events: [batch] - event types (0=censored, 1,2,...,K=event types)
        """
        return nll_competing_risk_hazard(phi, idx_durations, events,
                                        self.num_risks, self.reduction, self.training)


class CompetingRiskLoss(nn.Module):
    """
    Complete DySurv loss with competing risks support.

    Combines three loss components:
    1. Competing risk survival loss (cause-specific hazards)
    2. Autoencoder reconstruction loss
    3. KL-divergence regularization

    Args:
        alpha: List of 3 weights [survival_weight, ae_weight, kl_weight]
        num_risks: Number of competing event types
    """
    def __init__(self, alpha: list, num_risks: int = 2):
        super().__init__()
        self.alpha = alpha
        self.num_risks = num_risks
        self.loss_surv = NLLCompetingRiskLoss(num_risks=num_risks)
        self.loss_ae = nn.MSELoss()

    def forward(self, decoded, phi, mu, logvar, target_loghaz, target_ae):
        """
        Forward call with competing risk loss.

        Args:
            decoded: Reconstructed time-series from decoder
            phi: Survival predictions [batch, time_points, num_risks]
            mu: Latent mean from encoder
            logvar: Latent log variance from encoder
            target_loghaz: Tuple of (idx_durations, events)
            target_ae: Target time-series for reconstruction

        Returns:
            total_loss: Weighted combination of all loss components
        """
        # Unpack targets
        idx_durations, events = target_loghaz
        target_ae = target_ae[:, :, :-1].float()

        # Competing Risk Survival Loss
        loss_surv = self.loss_surv(phi, idx_durations, events)

        # Autoencoder Reconstruction Loss
        loss_ae = self.loss_ae(decoded, target_ae)

        # KL-Divergence Loss
        loss_kd = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / 10

        return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae + self.alpha[2] * loss_kd


######## Original Single Event Loss (kept for backwards compatibility) ########

class Loss(nn.Module):
    def __init__(self, alpha: list, pos_weight: float | None = None):
        super().__init__()
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss(pos_weight=pos_weight)
        self.loss_ae = nn.MSELoss()

    def forward(self, decoded, phi, mu, logvar, target_loghaz, target_ae):
        """
            Forward call of the Loss Module. Computes the DySurv model loss by combining three weighted losses.
                1. Survival loss: negative log likelihood logistic hazard or BCE loss over the predictions.
                2. AE loss: reconstruction or MSE loss
                3. KL-divergence: KL-divergence or pushing the model to have a latent space close to a normal distribution.
        """
        # Unpack data
        idx_durations, events = target_loghaz

        # Unpack targets
        target_ae = target_ae[:, :, :-1].float()

        # Survival Loss
        loss_surv = self.loss_surv(phi, idx_durations, events) # why divide by 10?

        # AutoEncoder Loss
        loss_ae = self.loss_ae(decoded, target_ae)/1

        # KL-divergence Loss
        loss_kd = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/10

        return self.alpha[0] * loss_surv + self.alpha[1] * loss_ae + self.alpha[2] * loss_kd

