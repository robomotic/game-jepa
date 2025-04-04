import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LatentEncoder(nn.Module):
    """Encoder module that transforms observations into latent representations."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [256, 512]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class PredictiveModel(nn.Module):
    """Predictive model that forecasts future latent states."""

    def __init__(self, latent_dim: int, context_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()
        layers = []
        prev_dim = latent_dim + context_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.predictor = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if len(latent.shape) == 1:
            latent = latent.unsqueeze(0)
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        combined = torch.cat([latent, context], dim=-1)
        return self.predictor(combined)

class JEPAAgent:
    """Agent based on JEPA architecture with predictive modeling and task-specific prediction."""

    def __init__(
        self,
        id: str,
        observation_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        context_dim: int = 64,
        learning_rate: float = 1e-4
    ):
        self.id = id
        self.observation_dim = observation_dim  # Window of past time steps
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Initialize models
        self.encoder = LatentEncoder(observation_dim, latent_dim)
        self.predictor = PredictiveModel(latent_dim, context_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict a single value
        )

        # Context representation
        self.context = torch.zeros(context_dim)

        # Initialize optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.prediction_optimizer = torch.optim.Adam(self.prediction_head.parameters(), lr=learning_rate)

        # Memory for experience
        self.experience_buffer = []
        self.max_buffer_size = 10000

        # Track last observation
        self.last_observation = None

        logger.info(f"Agent {id} initialized with observation_dim={observation_dim}, latent_dim={latent_dim}, context_dim={context_dim}")

    def update_context(self, new_info: torch.Tensor):
        """Update the agent's context vector with new information."""
        alpha = 0.8
        if new_info.shape != self.context.shape:
            if len(new_info.shape) > 1 and new_info.shape[0] == 1 and new_info.shape[1] == self.context.shape[0]:
                new_info = new_info.squeeze(0)
            elif len(new_info.shape) > 1:
                new_info = new_info[0]
        self.context = alpha * new_info + (1 - alpha) * self.context

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode an observation into a latent representation."""
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        return self.encoder(observation)

    def predict_next_value(self, observation: torch.Tensor) -> torch.Tensor:
        """Predict the next time series value based on observation."""
        latent = self.encode_observation(observation)
        prediction = self.prediction_head(latent)
        return prediction.squeeze()  # Return scalar or batch of scalars

    def train_step(self, current_obs: torch.Tensor, next_obs: torch.Tensor, next_value: torch.Tensor) -> Dict[str, float]:
        """Train on current observation, next observation, and next value."""
        if len(current_obs.shape) == 1:
            current_obs = current_obs.unsqueeze(0)
        if len(next_obs.shape) == 1:
            next_obs = next_obs.unsqueeze(0)
        if len(next_value.shape) == 0:
            next_value = next_value.unsqueeze(0)

        self.encoder_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()
        self.prediction_optimizer.zero_grad()

        # Encode observations
        current_latent = self.encoder(current_obs)
        next_latent_actual = self.encoder(next_obs)
        batch_size = current_latent.shape[0]
        context_batched = self.context.unsqueeze(0).expand(batch_size, -1)

        # Predict next latent and value
        next_latent_pred = self.predictor(current_latent, context_batched)
        value_pred = self.prediction_head(current_latent)

        # Losses
        prediction_loss = F.mse_loss(next_latent_pred, next_latent_actual.detach())
        value_loss = F.mse_loss(value_pred.squeeze(), next_value)  # Fixed shape mismatch
        if batch_size > 1:
            indices = torch.randperm(batch_size)
            negative_samples = next_latent_actual[indices]
            pos_similarity = F.cosine_similarity(next_latent_pred, next_latent_actual, dim=1)
            neg_similarity = F.cosine_similarity(next_latent_pred, negative_samples, dim=1)
            temperature = 0.1
            contrastive_loss = -torch.log(
                torch.exp(pos_similarity / temperature) /
                (torch.exp(pos_similarity / temperature) + torch.exp(neg_similarity / temperature))
            ).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=current_latent.device)

        total_loss = prediction_loss + contrastive_loss + value_loss
        total_loss.backward()

        self.encoder_optimizer.step()
        self.predictor_optimizer.step()
        self.prediction_optimizer.step()

        return {
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }

    def add_experience(self, current_obs: torch.Tensor, next_obs: torch.Tensor, next_value: torch.Tensor):
        """Add experience to buffer."""
        self.experience_buffer.append((current_obs, next_obs, next_value))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_from_buffer(self, batch_size: int = 32) -> Dict[str, float]:
        """Train on a batch from the experience buffer."""
        if len(self.experience_buffer) < batch_size:
            return {"error": "Not enough experiences for training"}

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        current_obs = torch.stack([item[0] for item in batch])
        next_obs = torch.stack([item[1] for item in batch])
        next_values = torch.stack([item[2] for item in batch])

        return self.train_step(current_obs, next_obs, next_values)

    def add_experience(self, current_obs: torch.Tensor, next_obs: torch.Tensor, next_value: torch.Tensor):
        """Add experience to buffer."""
        self.experience_buffer.append((current_obs, next_obs, next_value))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_from_buffer(self, batch_size: int = 32) -> Dict[str, float]:
        """Train on a batch from the experience buffer."""
        if len(self.experience_buffer) < batch_size:
            return {"error": "Not enough experiences for training"}

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        current_obs = torch.stack([item[0] for item in batch])
        next_obs = torch.stack([item[1] for item in batch])
        next_values = torch.stack([item[2] for item in batch])

        return self.train_step(current_obs, next_obs, next_values)