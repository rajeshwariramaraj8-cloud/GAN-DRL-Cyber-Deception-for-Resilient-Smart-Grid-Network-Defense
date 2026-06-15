"""
Neural Models for GAN-DRL Cyber Deception

This module contains the core learning components used by the reproducibility
pipeline:
  - GAN Generator for adversarial smart-grid traffic synthesis
  - GAN Discriminator for real/synthetic attack discrimination
  - DQN policy network for adaptive deception decisions
  - Replay buffer and DQN agent utilities
  - Checkpoint save/load helpers

The implementation is intentionally compact and deterministic so that reviewers
can execute the complete experiment using the repository scripts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


TensorBatch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_torch_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def activation_layer(name: str) -> nn.Module:
    normalized = str(name).lower().strip()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized in {"leaky_relu", "leakyrelu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.2)
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    output_activation: Optional[str] = None,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    previous_dim = int(input_dim)

    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, int(output_dim)))

    if output_activation is not None:
        layers.append(activation_layer(output_activation))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    """
    GAN generator.

    Input: latent vector z
    Output: synthetic attack feature vector scaled to [0, 1]
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.network = build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            dropout=dropout,
            output_activation="sigmoid",
            use_layer_norm=True,
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class Discriminator(nn.Module):
    """
    GAN discriminator.

    Input: attack feature vector
    Output: raw logit. BCEWithLogitsLoss should be used during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        activation: str = "leaky_relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.network = build_mlp(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            dropout=dropout,
            output_activation=None,
            use_layer_norm=False,
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).view(-1)


class DQN(nn.Module):
    """
    Deep Q-Network for adaptive cyber deception.

    Input: environment state vector
    Output: Q-values for all defense actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.network = build_mlp(
            input_dim=self.state_dim,
            hidden_dims=hidden_dims,
            output_dim=self.action_dim,
            activation=activation,
            dropout=dropout,
            output_activation=None,
            use_layer_norm=False,
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Fixed-size replay buffer for DQN training."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 42) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.rng = np.random.default_rng(seed)
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> TensorBatch:
        if len(self.buffer) < batch_size:
            raise ValueError("Replay buffer does not contain enough samples.")

        indices = self.rng.choice(len(self.buffer), size=int(batch_size), replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[int(index)] for index in indices))

        state_tensor = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device).view(-1, 1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1, 1)
        next_state_tensor = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device).view(-1, 1)

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor


@dataclass
class DQNUpdateResult:
    loss: float
    mean_q_value: float
    mean_target_q_value: float


class DQNAgent:
    """DQN agent with replay memory and target network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Iterable[int],
        config: Dict,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.device = device or get_device()
        self.seed = int(seed)

        dqn_cfg = config["dqn"]
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(dqn_cfg["gamma"])
        self.batch_size = int(dqn_cfg["batch_size"])
        self.target_update_interval = int(dqn_cfg["target_update_interval"])
        self.epsilon = float(dqn_cfg["epsilon_start"])
        self.epsilon_min = float(dqn_cfg["epsilon_min"])
        self.epsilon_decay = float(dqn_cfg["epsilon_decay"])
        self.update_counter = 0

        self.policy_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            dropout=0.0,
        ).to(self.device)

        self.target_net = DQN(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            dropout=0.0,
        ).to(self.device)

        self.hard_update_target()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(dqn_cfg["learning_rate"]))
        self.memory = ReplayBuffer(
            capacity=int(dqn_cfg["replay_memory_size"]),
            state_dim=state_dim,
            seed=seed,
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def hard_update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self, tau: float = 0.005) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_parameter.data.mul_(1.0 - tau)
                target_parameter.data.add_(tau * source_parameter.data)

    def update(self) -> Optional[DQNUpdateResult]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_interval == 0:
            self.hard_update_target()

        return DQNUpdateResult(
            loss=float(loss.item()),
            mean_q_value=float(current_q.detach().mean().item()),
            mean_target_q_value=float(target_q.detach().mean().item()),
        )

    def save(self, path: str | Path, extra: Optional[Dict] = None) -> None:
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_counter": self.update_counter,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "extra": extra or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load(self, path: str | Path, map_location: Optional[torch.device] = None) -> Dict:
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.update_counter = int(checkpoint.get("update_counter", 0))
        return checkpoint.get("extra", {})


def sample_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(int(batch_size), int(latent_dim), device=device)


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    return real_loss + fake_loss


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    target_labels = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, target_labels)


@torch.no_grad()
def generate_synthetic_attacks(
    generator: Generator,
    sample_count: int,
    latent_dim: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    generator.eval()
    generated_batches: List[np.ndarray] = []

    remaining = int(sample_count)
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        noise = sample_noise(current_batch, latent_dim, device)
        samples = generator(noise).detach().cpu().numpy()
        generated_batches.append(samples)
        remaining -= current_batch

    return np.vstack(generated_batches).astype(np.float32)


def create_gan_models(feature_dim: int, config: Dict, device: Optional[torch.device] = None) -> Tuple[Generator, Discriminator]:
    device = device or get_device()
    gan_cfg = config["gan"]

    generator = Generator(
        latent_dim=int(gan_cfg["latent_dim"]),
        output_dim=int(feature_dim),
        hidden_dims=list(gan_cfg["generator_hidden_dims"]),
        activation=str(gan_cfg["activation"]),
        dropout=float(gan_cfg["dropout"]),
    ).to(device)

    discriminator = Discriminator(
        input_dim=int(feature_dim),
        hidden_dims=list(gan_cfg["discriminator_hidden_dims"]),
        activation=str(gan_cfg["activation"]),
        dropout=float(gan_cfg["dropout"]),
    ).to(device)

    return generator, discriminator


def create_dqn_agent(config: Dict, device: Optional[torch.device] = None, seed: int = 42) -> DQNAgent:
    return DQNAgent(
        state_dim=int(config["dqn"]["state_dim"]),
        action_dim=len(config["dqn"]["action_space"]),
        hidden_dims=list(config["dqn"]["hidden_dims"]),
        config=config,
        device=device or get_device(),
        seed=seed,
    )


def save_gan_checkpoint(
    path: str | Path,
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    metadata: Optional[Dict] = None,
) -> None:
    checkpoint = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "metadata": metadata or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_gan_checkpoint(
    path: str | Path,
    generator: Generator,
    discriminator: Discriminator,
    generator_optimizer: Optional[torch.optim.Optimizer] = None,
    discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[torch.device] = None,
) -> Dict:
    checkpoint = torch.load(path, map_location=map_location or get_device())
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    if generator_optimizer is not None and "generator_optimizer" in checkpoint:
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])

    if discriminator_optimizer is not None and "discriminator_optimizer" in checkpoint:
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])

    return checkpoint.get("metadata", {})


def model_summary(config: Dict, feature_dim: int = 64) -> Dict:
    device = torch.device("cpu")
    generator, discriminator = create_gan_models(feature_dim=feature_dim, config=config, device=device)
    agent = create_dqn_agent(config=config, device=device, seed=int(config["reproducibility"]["global_seed"]))

    return {
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "dqn_policy_parameters": count_parameters(agent.policy_net),
        "gan_latent_dim": int(config["gan"]["latent_dim"]),
        "gan_feature_dim": int(feature_dim),
        "dqn_state_dim": int(config["dqn"]["state_dim"]),
        "dqn_action_dim": len(config["dqn"]["action_space"]),
        "action_space": list(config["dqn"]["action_space"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model summary for GAN-DQN cyber deception components.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--feature-dim", type=int, default=64, help="Attack feature dimension.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_torch_seed(
        seed=int(config["reproducibility"]["global_seed"]),
        deterministic=bool(config["reproducibility"].get("deterministic", True)),
    )

    summary = model_summary(config, feature_dim=args.feature_dim)
    result_dir = Path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "model_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Model summary saved to: {output_path}")


if __name__ == "__main__":
    main()
