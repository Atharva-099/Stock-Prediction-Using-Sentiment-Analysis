"""
Temporal Convolutional Network (TCN) for Stock Forecasting
=========================================================
Implementation of TCN architecture as recommended in research paper:
"entropy-25-00219-v2.pdf" - TCN ranked #1 for stock forecasting

Key Features:
- Dilated causal convolutions (no future information leakage)
- Residual connections (stable training)
- Multiple temporal resolutions (captures patterns at different scales)
- Highly parallelizable (faster than RNNs)

Architecture:
    Input → TCN Block 1 → TCN Block 2 → ... → Dense → Output
    Each block: Conv1D (dilated) → ReLU → Dropout → Conv1D → ReLU → Dropout → Residual
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import logging

logger = logging.getLogger(__name__)


class Chomp1d(nn.Module):
    """
    Removes the trailing padding from temporal convolutions
    Ensures causal (no future data leakage)
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single TCN temporal block with:
    - Dilated causal convolution
    - Weight normalization
    - ReLU activation
    - Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection (1x1 conv if dimensions don't match)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass with residual connection"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Full TCN network - stack of temporal blocks with increasing dilation
    
    Dilation increases exponentially: 1, 2, 4, 8, ...
    This allows the network to have a large receptive field while keeping
    the number of parameters manageable.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_inputs: Number of input features
            num_channels: List of hidden channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_inputs, seq_len)
        
        Returns:
            Output tensor of shape (batch_size, num_channels[-1], seq_len)
        """
        return self.network(x)


class TCNForecaster(nn.Module):
    """
    TCN model for stock price forecasting
    
    Architecture:
        Input features → TCN (multiple temporal blocks) → Dense → Output price
    """
    def __init__(self, input_size, hidden_channels=[64, 128, 64], 
                 kernel_size=3, dropout=0.2, output_size=1):
        """
        Args:
            input_size: Number of input features
            hidden_channels: List of hidden channel sizes (determines network depth)
            kernel_size: Convolutional kernel size
            dropout: Dropout rate
            output_size: Output dimension (1 for single price prediction)
        """
        super().__init__()
        
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Final linear layer
        self.linear = nn.Linear(hidden_channels[-1], output_size)
        
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        self.receptive_field = self._calculate_receptive_field(kernel_size, len(hidden_channels))
        
        logger.info(f"TCN Model initialized:")
        logger.info(f"  Input features: {input_size}")
        logger.info(f"  Hidden channels: {hidden_channels}")
        logger.info(f"  Kernel size: {kernel_size}")
        logger.info(f"  Dropout: {dropout}")
        logger.info(f"  Receptive field: {self.receptive_field}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               OR (batch_size, input_size) for single timestep
        
        Returns:
            Predicted price(s) of shape (batch_size, output_size)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # (batch_size, input_size) → (batch_size, input_size, 1)
            x = x.unsqueeze(-1)
        else:
            # (batch_size, seq_len, input_size) → (batch_size, input_size, seq_len)
            x = x.transpose(1, 2)
        
        # TCN forward pass
        y = self.tcn(x)  # (batch_size, hidden_channels[-1], seq_len)
        
        # Take the last timestep
        y = y[:, :, -1]  # (batch_size, hidden_channels[-1])
        
        # Final linear layer
        output = self.linear(y)  # (batch_size, output_size)
        
        return output
    
    def _calculate_receptive_field(self, kernel_size, num_levels):
        """
        Calculate the receptive field of the TCN
        
        Receptive field = sum of (kernel_size - 1) * dilation for each level
        """
        receptive_field = 1
        for i in range(num_levels):
            dilation = 2 ** i
            receptive_field += (kernel_size - 1) * dilation
        return receptive_field
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TCNForecasterMultiHorizon(nn.Module):
    """
    TCN model for multi-horizon forecasting (1-day, 7-day, 14-day)
    
    Uses shared TCN backbone with separate prediction heads
    """
    def __init__(self, input_size, hidden_channels=[64, 128, 64], 
                 kernel_size=3, dropout=0.2, horizons=[1, 7, 14]):
        """
        Args:
            input_size: Number of input features
            hidden_channels: List of hidden channel sizes
            kernel_size: Convolutional kernel size
            dropout: Dropout rate
            horizons: List of prediction horizons (days ahead)
        """
        super().__init__()
        
        # Shared TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Separate prediction heads for each horizon
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(hidden_channels[-1], 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
            for h in horizons
        })
        
        self.horizons = horizons
        self.input_size = input_size
        
        logger.info(f"Multi-Horizon TCN initialized:")
        logger.info(f"  Horizons: {horizons}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")

    def forward(self, x, horizon=None):
        """
        Args:
            x: Input tensor
            horizon: Specific horizon to predict (None = all horizons)
        
        Returns:
            Dictionary of predictions by horizon, or single prediction
        """
        # Handle input dimensions
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        else:
            x = x.transpose(1, 2)
        
        # Shared TCN features
        features = self.tcn(x)
        features = features[:, :, -1]
        
        # Predictions for each horizon
        if horizon is not None:
            # Single horizon
            return self.prediction_heads[f'horizon_{horizon}'](features)
        else:
            # All horizons
            return {
                h: self.prediction_heads[f'horizon_{h}'](features)
                for h in self.horizons
            }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tcn_model(input_size, model_type='standard', **kwargs):
    """
    Factory function to create TCN models
    
    Args:
        input_size: Number of input features
        model_type: 'standard' or 'multi_horizon'
        **kwargs: Additional arguments for the model
    
    Returns:
        TCN model instance
    """
    if model_type == 'standard':
        return TCNForecaster(input_size, **kwargs)
    elif model_type == 'multi_horizon':
        return TCNForecasterMultiHorizon(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == '__main__':
    # Test TCN model
    print("Testing TCN Model...")
    
    # Standard model
    model = TCNForecaster(
        input_size=20,
        hidden_channels=[64, 128, 64],
        kernel_size=3,
        dropout=0.2
    )
    
    # Test forward pass
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 20)
    
    output = model(x)
    print(f"✓ Standard TCN output shape: {output.shape}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Multi-horizon model
    model_mh = TCNForecasterMultiHorizon(
        input_size=20,
        hidden_channels=[64, 128, 64],
        horizons=[1, 7, 14]
    )
    
    outputs = model_mh(x)
    print(f"\n✓ Multi-Horizon TCN:")
    for horizon, pred in outputs.items():
        print(f"  {horizon}: {pred.shape}")
    print(f"  Parameters: {model_mh.count_parameters():,}")



