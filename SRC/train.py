"""
Training Pipeline for GAN-DRL Cyber Deception

This script trains the two learning modules used in the reproducibility package:
  1. GAN for adversarial smart-grid traffic synthesis
  2. DQN for adaptive cyber deception and resilient response

Expected execution:
    python src/train.py --config config.yaml

The script writes all model checkpoints, synthetic attack samples, and training
logs under the paths defined in config.yaml.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import (
    DQNAgent,
    create_dqn_agent,
    create_gan_models,
    discriminator_loss,
    generate_synthetic_attacks,
    generator_loss,
    get_device,
    load_config,
    sample_noise,
    save_gan_checkpoint,
    set_torch_seed,
)
from smartgrid_env import SmartGridCyberEnv


EPS = 1e-9


def ensure_directories(config: Dict) -> Dict[str, Path]:
    paths = {
        "generated": Path(config["paths"]["generated_data_dir"]),
        "models": Path(config["paths"]["model_dir"]),
        "results": Path(config["paths"]["result_dir"]),
        "figures": Path(config["paths"]["figure_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_feature_schema(config: Dict) -> List[str]:
    schema_path = Path(config["paths"]["generated_data_dir"]) / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            "feature_schema.json was not found. Generate the dataset first:\n"
            "  python src/data_pipeline.py --config config.yaml"
        )

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return list(schema["feature_columns"])


def load_training_dataset(config: Dict) -> pd.DataFrame:
    train_path = Path(config["paths"]["generated_data_dir"]) / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "train.csv was not found. Generate the dataset first:\n"
            "  python src/data_pipeline.py --config config.yaml"
        )
    return pd.read_csv(train_path)


def prepare_attack_tensor(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Path,
) -> Tuple[torch.Tensor, MinMaxScaler]:
    attack_df = train_df[train_df["label"] == 1].copy()
    if attack_df.empty:
        raise ValueError("No adversarial samples were found in the training dataset.")

    real_attack_features = attack_df[feature_cols].astype(float).to_numpy(dtype=np.float32)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(real_attack_features).astype(np.float32)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return torch.as_tensor(scaled_features, dtype=torch.float32), scaler


def histogram_kl(real_values: np.ndarray, generated_values: np.ndarray, bins: int = 30) -> float:
    real_hist, bin_edges = np.histogram(real_values, bins=bins, density=False)
    gen_hist, _ = np.histogram(generated_values, bins=bin_edges, density=False)

    real_hist = real_hist.astype(float) + EPS
    gen_hist = gen_hist.astype(float) + EPS

    real_hist = real_hist / real_hist.sum()
    gen_hist = gen_hist / gen_hist.sum()

    return float(np.sum(real_hist * np.log(real_hist / gen_hist)))


def estimate_average_kl(real_scaled: np.ndarray, generated_scaled: np.ndarray) -> float:
    if real_scaled.shape[1] == 0:
        return 0.0

    kl_values = []
    selected_features = min(12, real_scaled.shape[1])
    for index in range(selected_features):
        kl_values.append(histogram_kl(real_scaled[:, index], generated_scaled[:, index]))

    return float(np.mean(kl_values))


def train_gan(
    config: Dict,
    train_df: pd.DataFrame,
    feature_cols: List[str],
    paths: Dict[str, Path],
    device: torch.device,
) -> np.ndarray:
    gan_cfg = config["gan"]

    scaler_path = paths["models"] / "attack_feature_scaler.joblib"
    attack_tensor, scaler = prepare_attack_tensor(train_df, feature_cols, scaler_path)
    feature_dim = attack_tensor.shape[1]

    dataset = TensorDataset(attack_tensor)
    data_loader = DataLoader(
        dataset,
        batch_size=int(gan_cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    generator, discriminator = create_gan_models(feature_dim=feature_dim, config=config, device=device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(gan_cfg["learning_rate"]),
        betas=(float(gan_cfg["beta1"]), float(gan_cfg["beta2"])),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(gan_cfg["learning_rate"]),
        betas=(float(gan_cfg["beta1"]), float(gan_cfg["beta2"])),
    )

    gan_log = []
    real_scaled_np = attack_tensor.numpy()
    start_time = time.time()

    print("Training GAN adversarial traffic generator...")
    for epoch in tqdm(range(1, int(gan_cfg["epochs"]) + 1), desc="GAN epochs"):
        generator.train()
        discriminator.train()

        epoch_d_loss = []
        epoch_g_loss = []

        for (real_batch,) in data_loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            discriminator_optimizer.zero_grad(set_to_none=True)

            noise = sample_noise(batch_size, int(gan_cfg["latent_dim"]), device)
            fake_batch = generator(noise).detach()

            real_logits = discriminator(real_batch)
            fake_logits = discriminator(fake_batch)

            d_loss = discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0)
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)

            noise = sample_noise(batch_size, int(gan_cfg["latent_dim"]), device)
            generated_batch = generator(noise)
            generated_logits = discriminator(generated_batch)

            g_loss = generator_loss(generated_logits)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            generator_optimizer.step()

            epoch_d_loss.append(float(d_loss.item()))
            epoch_g_loss.append(float(g_loss.item()))

        generated_probe = generate_synthetic_attacks(
            generator=generator,
            sample_count=min(1000, int(gan_cfg["synthetic_attack_samples"])),
            latent_dim=int(gan_cfg["latent_dim"]),
            device=device,
            batch_size=512,
        )
        average_kl = estimate_average_kl(real_scaled_np, generated_probe)

        gan_log.append(
            {
                "epoch": epoch,
                "discriminator_loss": float(np.mean(epoch_d_loss)),
                "generator_loss": float(np.mean(epoch_g_loss)),
                "average_kl_divergence": average_kl,
            }
        )

    synthetic_scaled = generate_synthetic_attacks(
        generator=generator,
        sample_count=int(gan_cfg["synthetic_attack_samples"]),
        latent_dim=int(gan_cfg["latent_dim"]),
        device=device,
        batch_size=512,
    )

    synthetic_original = scaler.inverse_transform(synthetic_scaled)
    synthetic_df = pd.DataFrame(synthetic_original, columns=feature_cols)
    synthetic_df["label"] = 1
    synthetic_df["attack_type"] = "gan_generated"

    gan_log_df = pd.DataFrame(gan_log)
    gan_log_path = paths["results"] / "gan_training_log.csv"
    gan_log_df.to_csv(gan_log_path, index=False)

    synthetic_csv_path = paths["generated"] / "gan_synthetic_attacks.csv"
    synthetic_df.to_csv(synthetic_csv_path, index=False)

    synthetic_npy_path = paths["models"] / "generated_attack_samples.npy"
    np.save(synthetic_npy_path, synthetic_scaled)

    checkpoint_path = paths["models"] / "gan_checkpoint.pt"
    save_gan_checkpoint(
        checkpoint_path,
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        metadata={
            "feature_dim": int(feature_dim),
            "feature_columns": feature_cols,
            "latent_dim": int(gan_cfg["latent_dim"]),
            "synthetic_attack_samples": int(gan_cfg["synthetic_attack_samples"]),
            "training_seconds": round(time.time() - start_time, 4),
            "final_average_kl_divergence": float(gan_log[-1]["average_kl_divergence"]),
        },
    )

    print(f"GAN checkpoint saved to: {checkpoint_path}")
    print(f"GAN training log saved to: {gan_log_path}")
    print(f"Synthetic attacks saved to: {synthetic_csv_path}")

    return synthetic_scaled


def train_dqn(
    config: Dict,
    paths: Dict[str, Path],
    generated_attack_samples: np.ndarray,
    device: torch.device,
) -> DQNAgent:
    dqn_cfg = config["dqn"]
    seed = int(config["reproducibility"]["global_seed"])

    env = SmartGridCyberEnv(
        config=config,
        seed=seed,
        use_gan=True,
        use_deception=True,
        use_dim=True,
        generated_attack_samples=generated_attack_samples,
        max_steps=int(dqn_cfg["max_steps_per_episode"]),
    )

    agent = create_dqn_agent(config=config, device=device, seed=seed)

    episode_rows = []
    update_rows = []
    best_reward = -float("inf")
    start_time = time.time()

    print("Training DQN adaptive cyber deception agent...")
    for episode in tqdm(range(1, int(dqn_cfg["episodes"]) + 1), desc="DQN episodes"):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_losses = []

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            update_result = agent.update()
            if update_result is not None:
                episode_losses.append(update_result.loss)
                if episode % 25 == 0:
                    update_rows.append(
                        {
                            "episode": episode,
                            "step": info["step"],
                            "loss": update_result.loss,
                            "mean_q_value": update_result.mean_q_value,
                            "mean_target_q_value": update_result.mean_target_q_value,
                            "epsilon": agent.epsilon,
                        }
                    )

            state = next_state
            episode_reward += float(reward)

        agent.decay_epsilon()
        summary = env.summarize_episode()

        row = {
            "episode": episode,
            "episode_reward": episode_reward,
            "average_step_reward": episode_reward / max(1, int(dqn_cfg["max_steps_per_episode"])),
            "epsilon": agent.epsilon,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else np.nan,
            "mttc": summary["mttc"],
            "fpr": summary["fpr"],
            "rrl": summary["rrl"],
            "dus": summary["dus"],
            "attack_surface_reduction": summary["attack_surface_reduction"],
            "final_resilience": summary["final_resilience"],
            "successful_compromises": summary["successful_compromises"],
            "deceptive_captures": summary["deceptive_captures"],
        }
        episode_rows.append(row)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(
                paths["models"] / "best_dqn_checkpoint.pt",
                extra={
                    "best_episode": episode,
                    "best_episode_reward": best_reward,
                    "summary": summary,
                },
            )

    agent.save(
        paths["models"] / "final_dqn_checkpoint.pt",
        extra={
            "episodes": int(dqn_cfg["episodes"]),
            "training_seconds": round(time.time() - start_time, 4),
            "final_epsilon": agent.epsilon,
        },
    )

    episode_log = pd.DataFrame(episode_rows)
    update_log = pd.DataFrame(update_rows)

    episode_log_path = paths["results"] / "dqn_training_log.csv"
    update_log_path = paths["results"] / "dqn_update_log.csv"

    episode_log.to_csv(episode_log_path, index=False)
    update_log.to_csv(update_log_path, index=False)

    print(f"Best DQN checkpoint saved to: {paths['models'] / 'best_dqn_checkpoint.pt'}")
    print(f"Final DQN checkpoint saved to: {paths['models'] / 'final_dqn_checkpoint.pt'}")
    print(f"DQN episode log saved to: {episode_log_path}")
    print(f"DQN update log saved to: {update_log_path}")

    return agent


def write_training_summary(
    config: Dict,
    paths: Dict[str, Path],
    feature_cols: List[str],
    generated_attack_samples: np.ndarray,
    device: torch.device,
) -> None:
    gan_log_path = paths["results"] / "gan_training_log.csv"
    dqn_log_path = paths["results"] / "dqn_training_log.csv"

    gan_log = pd.read_csv(gan_log_path) if gan_log_path.exists() else pd.DataFrame()
    dqn_log = pd.read_csv(dqn_log_path) if dqn_log_path.exists() else pd.DataFrame()

    summary = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "device": str(device),
        "feature_count": len(feature_cols),
        "generated_attack_samples": int(len(generated_attack_samples)),
        "gan_epochs": int(config["gan"]["epochs"]),
        "dqn_episodes": int(config["dqn"]["episodes"]),
        "dqn_max_steps_per_episode": int(config["dqn"]["max_steps_per_episode"]),
        "final_gan_generator_loss": None if gan_log.empty else float(gan_log.iloc[-1]["generator_loss"]),
        "final_gan_discriminator_loss": None if gan_log.empty else float(gan_log.iloc[-1]["discriminator_loss"]),
        "final_gan_average_kl_divergence": None if gan_log.empty else float(gan_log.iloc[-1]["average_kl_divergence"]),
        "best_dqn_episode_reward": None if dqn_log.empty else float(dqn_log["episode_reward"].max()),
        "final_dqn_episode_reward": None if dqn_log.empty else float(dqn_log.iloc[-1]["episode_reward"]),
        "final_dqn_epsilon": None if dqn_log.empty else float(dqn_log.iloc[-1]["epsilon"]),
        "output_files": {
            "gan_checkpoint": str(paths["models"] / "gan_checkpoint.pt"),
            "best_dqn_checkpoint": str(paths["models"] / "best_dqn_checkpoint.pt"),
            "final_dqn_checkpoint": str(paths["models"] / "final_dqn_checkpoint.pt"),
            "generated_attack_samples": str(paths["models"] / "generated_attack_samples.npy"),
            "gan_training_log": str(gan_log_path),
            "dqn_training_log": str(dqn_log_path),
        },
    }

    summary_path = paths["results"] / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training summary saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAN and DQN modules for cyber deception.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument(
        "--skip-gan",
        action="store_true",
        help="Skip GAN training and reuse results/models/generated_attack_samples.npy.",
    )
    parser.add_argument(
        "--skip-dqn",
        action="store_true",
        help="Skip DQN training after GAN sample preparation.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["reproducibility"]["global_seed"])
    deterministic = bool(config["reproducibility"].get("deterministic", True))

    set_torch_seed(seed, deterministic=deterministic)
    random.seed(seed)
    np.random.seed(seed)

    paths = ensure_directories(config)
    device = get_device(prefer_cuda=True)

    print(f"Using device: {device}")

    feature_cols = load_feature_schema(config)
    train_df = load_training_dataset(config)

    if args.skip_gan:
        sample_path = paths["models"] / "generated_attack_samples.npy"
        if not sample_path.exists():
            raise FileNotFoundError(
                "Cannot skip GAN training because generated_attack_samples.npy does not exist."
            )
        generated_attack_samples = np.load(sample_path)
        print(f"Loaded existing GAN samples from: {sample_path}")
    else:
        generated_attack_samples = train_gan(
            config=config,
            train_df=train_df,
            feature_cols=feature_cols,
            paths=paths,
            device=device,
        )

    if not args.skip_dqn:
        train_dqn(
            config=config,
            paths=paths,
            generated_attack_samples=generated_attack_samples,
            device=device,
        )

    write_training_summary(
        config=config,
        paths=paths,
        feature_cols=feature_cols,
        generated_attack_samples=generated_attack_samples,
        device=device,
    )

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
