"""
Transfer Learning Model for Financial Forecasting
==================================================
Neural network model with:
- Freezable backbone (pre-trained on historical data 1999-2022)
- Fine-tunable head (adapted to recent data 2023-2025)

Supports multiple architectures:
- TCN (Temporal Convolutional Network)
- LSTM, BiLSTM
- Transformer
- CNN-LSTM Hybrid

Author: CMU Financial Forecasting Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """Single temporal block for TCN"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """
    Temporal Convolutional Network Backbone
    Pre-trained on historical data, can be frozen during fine-tuning
    """
    
    def __init__(self, input_size, hidden_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_channels = hidden_channels
        
        layers = []
        num_channels = [input_size] + hidden_channels
        
        for i in range(len(hidden_channels)):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    num_channels[i], num_channels[i+1],
                    kernel_size, stride=1, dilation=dilation,
                    padding=padding, dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        self.output_size = hidden_channels[-1]
    
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Return last timestep: (batch, hidden)
        return out[:, :, -1]


class LSTMBackbone(nn.Module):
    """LSTM Backbone"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_size = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # Last timestep


class TransformerBackbone(nn.Module):
    """Transformer Backbone"""
    
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_size = d_model
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return x[:, -1, :]  # Last timestep


class CNNLSTMBackbone(nn.Module):
    """CNN-LSTM Hybrid Backbone"""
    
    def __init__(self, input_size, cnn_channels=32, lstm_hidden=64, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True)
        
        self.output_size = lstm_hidden
    
    def forward(self, x):
        # CNN expects (batch, channels, seq)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = self.dropout(x)
        
        # LSTM expects (batch, seq, features)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return out[:, -1, :]


class FineTunableHead(nn.Module):
    """
    Fine-tunable head for transfer learning
    This is the part that gets fine-tuned on recent data
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransferLearningModel(nn.Module):
    """
    Complete Transfer Learning Model
    
    Architecture:
    - Backbone: Pre-trained on historical data (1999-2022)
    - Head: Fine-tuned on recent data (2023-2025)
    
    Training Strategy:
    1. Pre-train entire model on historical data
    2. Freeze backbone
    3. Fine-tune head on recent data
    """
    
    def __init__(self, 
                 input_size,
                 backbone_type='tcn',
                 backbone_config=None,
                 head_config=None):
        super().__init__()
        
        self.input_size = input_size
        self.backbone_type = backbone_type
        
        # Default configs
        backbone_config = backbone_config or {}
        head_config = head_config or {}
        
        # Create backbone
        if backbone_type == 'tcn':
            self.backbone = TCNBackbone(
                input_size,
                hidden_channels=backbone_config.get('hidden_channels', [64, 128, 64]),
                kernel_size=backbone_config.get('kernel_size', 3),
                dropout=backbone_config.get('dropout', 0.2)
            )
        elif backbone_type == 'lstm':
            self.backbone = LSTMBackbone(
                input_size,
                hidden_size=backbone_config.get('hidden_size', 128),
                num_layers=backbone_config.get('num_layers', 2),
                dropout=backbone_config.get('dropout', 0.2),
                bidirectional=False
            )
        elif backbone_type == 'bilstm':
            self.backbone = LSTMBackbone(
                input_size,
                hidden_size=backbone_config.get('hidden_size', 128),
                num_layers=backbone_config.get('num_layers', 2),
                dropout=backbone_config.get('dropout', 0.2),
                bidirectional=True
            )
        elif backbone_type == 'transformer':
            self.backbone = TransformerBackbone(
                input_size,
                d_model=backbone_config.get('d_model', 64),
                nhead=backbone_config.get('nhead', 4),
                num_layers=backbone_config.get('num_layers', 2),
                dropout=backbone_config.get('dropout', 0.1)
            )
        elif backbone_type == 'cnn_lstm':
            self.backbone = CNNLSTMBackbone(
                input_size,
                cnn_channels=backbone_config.get('cnn_channels', 32),
                lstm_hidden=backbone_config.get('lstm_hidden', 64),
                dropout=backbone_config.get('dropout', 0.2)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        # Create head
        self.head = FineTunableHead(
            input_size=self.backbone.output_size,
            hidden_sizes=head_config.get('hidden_sizes', [64, 32]),
            output_size=head_config.get('output_size', 1),
            dropout=head_config.get('dropout', 0.2)
        )
        
        self._backbone_frozen = False
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        logger.info("✓ Backbone frozen - only head will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        logger.info("✓ Backbone unfrozen - full model will be trained")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def save_backbone(self, filepath):
        """Save backbone weights"""
        torch.save(self.backbone.state_dict(), filepath)
        logger.info(f"✓ Backbone saved to: {filepath}")
    
    def load_backbone(self, filepath):
        """Load pre-trained backbone weights"""
        self.backbone.load_state_dict(torch.load(filepath))
        logger.info(f"✓ Backbone loaded from: {filepath}")
    
    def save_full_model(self, filepath):
        """Save complete model"""
        torch.save({
            'backbone': self.backbone.state_dict(),
            'head': self.head.state_dict(),
            'config': {
                'input_size': self.input_size,
                'backbone_type': self.backbone_type
            }
        }, filepath)
        logger.info(f"✓ Full model saved to: {filepath}")
    
    def load_full_model(self, filepath):
        """Load complete model"""
        checkpoint = torch.load(filepath)
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.head.load_state_dict(checkpoint['head'])
        logger.info(f"✓ Full model loaded from: {filepath}")


class TransferLearningTrainer:
    """
    Trainer for transfer learning approach
    
    Training pipeline:
    1. Pre-train on historical data (train backbone + head)
    2. Fine-tune on recent data (train head only)
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        self.training_history = {
            'pretrain': {'train_loss': [], 'val_loss': []},
            'finetune': {'train_loss': [], 'val_loss': []}
        }
    
    def prepare_sequences(self, X, y, seq_length=10):
        """Convert data to sequences for time series modeling"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def pretrain(self, X_train, y_train, X_val=None, y_val=None,
                 epochs=100, batch_size=32, lr=0.001, patience=15,
                 seq_length=10):
        """
        Pre-train the full model on historical data
        """
        logger.info("=" * 80)
        logger.info("PRE-TRAINING ON HISTORICAL DATA")
        logger.info("=" * 80)
        
        # Ensure backbone is unfrozen
        self.model.unfreeze_backbone()
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.prepare_sequences(X_train_scaled, y_train_scaled, seq_length)
        
        logger.info(f"Training samples: {len(X_seq)}")
        logger.info(f"Sequence length: {seq_length}")
        logger.info(f"Features: {X_seq.shape[2]}")
        logger.info(f"Total params: {self.model.get_total_params():,}")
        logger.info(f"Trainable params: {self.model.get_trainable_params():,}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled, seq_length)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        else:
            X_val_tensor = None
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            permutation = torch.randperm(X_tensor.size(0))
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, X_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.training_history['pretrain']['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                self.training_history['pretrain']['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}")
        
        logger.info("✓ Pre-training complete")
        return self.training_history['pretrain']
    
    def finetune(self, X_train, y_train, X_val=None, y_val=None,
                 epochs=50, batch_size=16, lr=0.0005, patience=10,
                 seq_length=10):
        """
        Fine-tune only the head on recent data
        """
        logger.info("=" * 80)
        logger.info("FINE-TUNING HEAD ON RECENT DATA")
        logger.info("=" * 80)
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        logger.info(f"Total params: {self.model.get_total_params():,}")
        logger.info(f"Trainable params (head only): {self.model.get_trainable_params():,}")
        
        # Scale data (using fitted scalers)
        X_train_scaled = self.scaler_X.transform(X_train)
        y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.prepare_sequences(X_train_scaled, y_train_scaled, seq_length)
        
        logger.info(f"Fine-tuning samples: {len(X_seq)}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        # Validation
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled, seq_length)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
        else:
            X_val_tensor = None
        
        # Only optimize head parameters
        optimizer = torch.optim.Adam(self.model.head.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch
            permutation = torch.randperm(X_tensor.size(0))
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, X_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_X = X_tensor[indices]
                batch_y = y_tensor[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self.training_history['finetune']['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                self.training_history['finetune']['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")
            else:
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}")
        
        logger.info("✓ Fine-tuning complete")
        return self.training_history['finetune']
    
    def predict(self, X, seq_length=10):
        """Make predictions"""
        self.model.eval()
        
        X_scaled = self.scaler_X.transform(X)
        X_seq, _ = self.prepare_sequences(X_scaled, np.zeros(len(X)), seq_length)
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
        
        predictions = self.scaler_y.inverse_transform(pred_scaled)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test, seq_length=10):
        """Evaluate model and return metrics"""
        predictions = self.predict(X_test, seq_length)
        
        # Align predictions with actual values
        y_actual = y_test[seq_length:]
        
        # Compute metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, predictions)
        mape = np.mean(np.abs((y_actual - predictions) / (y_actual + 1e-8))) * 100
        r2 = r2_score(y_actual, predictions)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'mse': mse
        }
        
        return metrics, predictions, y_actual
    
    def save(self, filepath):
        """Save trainer state"""
        state = {
            'model': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'history': self.training_history
        }
        torch.save(state, filepath)
        logger.info(f"✓ Trainer saved to: {filepath}")
    
    def load(self, filepath):
        """Load trainer state"""
        state = torch.load(filepath)
        self.model.load_state_dict(state['model'])
        self.scaler_X = state['scaler_X']
        self.scaler_y = state['scaler_y']
        self.training_history = state['history']
        logger.info(f"✓ Trainer loaded from: {filepath}")


def create_model(input_size, backbone_type='tcn', **kwargs):
    """Factory function to create transfer learning model"""
    return TransferLearningModel(
        input_size=input_size,
        backbone_type=backbone_type,
        backbone_config=kwargs.get('backbone_config'),
        head_config=kwargs.get('head_config')
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test model creation
    model = create_model(input_size=20, backbone_type='tcn')
    print(f"Model created: {model.backbone_type}")
    print(f"Total params: {model.get_total_params():,}")
    
    # Test forward pass
    x = torch.randn(32, 10, 20)  # batch=32, seq=10, features=20
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test freeze/unfreeze
    model.freeze_backbone()
    print(f"Trainable after freeze: {model.get_trainable_params():,}")
    
    model.unfreeze_backbone()
    print(f"Trainable after unfreeze: {model.get_trainable_params():,}")


