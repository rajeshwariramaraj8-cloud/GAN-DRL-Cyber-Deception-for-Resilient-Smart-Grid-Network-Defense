"""
Smart-Grid Cyber Deception Environment

This module implements the reusable simulation environment for the GAN-DQN
cyber deception pipeline. It models:
  - Smart-grid SCADA-IP topology
  - Benign, compromised, and deceptive node states
  - Attack propagation under adversarial pressure
  - Deception actions, traffic rerouting, quarantine, and honeypot activation
  - Digital Immune Mechanism based self-healing
  - Reward computation for DQN training
  - Runtime metrics used for evaluation

The environment is intentionally lightweight and does not depend on OpenAI Gym.
It exposes the familiar reset() and step(action) interface so that it can be
used directly by train.py and evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml


EPS = 1e-9


BENIGN = 0
COMPROMISED = 1
DECEPTIVE = 2
QUARANTINED = 3


@dataclass
class StepMetrics:
    total_attacks: int = 0
    successful_compromises: int = 0
    deceptive_captures: int = 0
    quarantines: int = 0
    reroutes: int = 0
    false_positives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    service_disruptions: int = 0
    recovered_nodes: int = 0


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def entropy_from_distribution(values: Iterable[float]) -> float:
    probs = np.asarray(list(values), dtype=float)
    if probs.size == 0:
        return 0.0
    probs = probs + EPS
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)) / np.log(len(probs) + EPS))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class SmartGridCyberEnv:
    """
    Lightweight cyber-physical smart-grid defense environment.

    Parameters
    ----------
    config:
        Loaded YAML configuration dictionary.
    seed:
        Random seed for deterministic execution.
    dataset_path:
        Optional path to generated traffic CSV. If not supplied, the full
        generated dataset path from config is used.
    use_gan:
        Whether GAN-enhanced adversarial pressure is considered active.
    use_deception:
        Whether deception actions and decoy states are active.
    use_dim:
        Whether the Digital Immune Mechanism is active.
    generated_attack_samples:
        Optional generated attack feature matrix produced by the GAN. When
        supplied, samples are injected into the state dynamics as adversarial
        pressure.
    max_steps:
        Optional episode length. Defaults to config dqn.max_steps_per_episode.
    """

    def __init__(
        self,
        config: Dict,
        seed: int = 42,
        dataset_path: Optional[str | Path] = None,
        use_gan: bool = True,
        use_deception: bool = True,
        use_dim: bool = True,
        generated_attack_samples: Optional[np.ndarray] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = set_seed(self.seed)

        self.paths = config["paths"]
        self.generated_dir = Path(self.paths["generated_data_dir"])
        self.dataset_path = Path(dataset_path) if dataset_path else self.generated_dir / "smartgrid_scada_ip_dataset.csv"
        self.nodes_path = self.generated_dir / "smartgrid_nodes.csv"
        self.edges_path = self.generated_dir / "smartgrid_edges.csv"

        self._validate_required_files()

        self.node_df = pd.read_csv(self.nodes_path)
        self.edge_df = pd.read_csv(self.edges_path)
        self.dataset = pd.read_csv(self.dataset_path)

        self.graph = self._build_graph()
        self.node_count = int(config["smartgrid"]["nodes"])
        self.action_names = list(config["dqn"]["action_space"])
        self.action_size = len(self.action_names)
        self.state_dim = int(config["dqn"]["state_dim"])
        self.max_steps = int(max_steps or config["dqn"]["max_steps_per_episode"])

        self.use_gan = bool(use_gan)
        self.use_deception = bool(use_deception)
        self.use_dim = bool(use_dim)
        self.generated_attack_samples = generated_attack_samples

        self.feature_cols = self._feature_columns()
        self.feature_means = self.dataset[self.feature_cols].mean(numeric_only=True)
        self.feature_stds = self.dataset[self.feature_cols].std(numeric_only=True).replace(0.0, 1.0)

        self.reward_cfg = config["reward"]
        self.dim_cfg = config["digital_immune_mechanism"]

        self.node_states = np.zeros(self.node_count, dtype=np.int64)
        self.compromise_start_time = np.full(self.node_count, -1.0, dtype=float)
        self.recovery_latencies: List[float] = []

        self.cursor = 0
        self.step_id = 0
        self.episode_id = 0
        self.done = False

        self.deception_level = 0.10 if self.use_deception else 0.0
        self.service_quality = 0.92
        self.resilience_potential = 0.78
        self.attack_pressure = 0.10
        self.previous_attack_pressure = 0.10
        self.deception_freshness = 0.50
        self.adaptation_momentum = 0.0

        self.recent_rewards = deque(maxlen=100)
        self.recent_attack_pressure = deque(maxlen=100)
        self.recent_deception_success = deque(maxlen=100)
        self.memory_bank = deque(maxlen=int(self.dim_cfg["memory_bank_size"]))

        self.metrics = StepMetrics()
        self.attack_confidence_scores: List[float] = []
        self.attack_labels: List[int] = []
        self.resilience_trace: List[float] = []
        self.reward_trace: List[float] = []
        self.deception_trace: List[float] = []
        self.compromise_trace: List[int] = []
        self.action_trace: List[str] = []

    def _validate_required_files(self) -> None:
        missing = [str(path) for path in [self.dataset_path, self.nodes_path, self.edges_path] if not path.exists()]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(
                "Generated dataset files are missing. Run the dataset generator first:\n"
                "  python src/data_pipeline.py --config config.yaml\n"
                f"Missing files:\n  - {joined}"
            )

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        for _, row in self.node_df.iterrows():
            graph.add_node(
                int(row["node_id"]),
                role=str(row["role"]),
                zone=int(row["zone"]),
                criticality=safe_float(row["criticality"], 0.5),
                vulnerability=safe_float(row["baseline_vulnerability"], 0.4),
            )

        for _, row in self.edge_df.iterrows():
            graph.add_edge(
                int(row["source"]),
                int(row["target"]),
                bandwidth_mbps=safe_float(row["bandwidth_mbps"], 100.0),
                latency_ms=safe_float(row["latency_ms"], 18.0),
                link_type=str(row.get("link_type", "telemetry")),
            )
        return graph

    def _feature_columns(self) -> List[str]:
        excluded = {
            "timestamp_id",
            "source_node",
            "destination_node",
            "source_role",
            "destination_role",
            "source_zone",
            "destination_zone",
            "protocol",
            "attack_type",
            "label",
        }
        return [col for col in self.dataset.columns if col not in excluded]

    def reset(self) -> np.ndarray:
        self.episode_id += 1
        self.step_id = 0
        self.done = False

        self.cursor = int(self.rng.integers(0, max(1, len(self.dataset) - self.max_steps)))
        self.node_states = np.zeros(self.node_count, dtype=np.int64)
        self.compromise_start_time = np.full(self.node_count, -1.0, dtype=float)
        self.recovery_latencies = []

        self.deception_level = 0.10 if self.use_deception else 0.0
        self.service_quality = 0.92
        self.resilience_potential = 0.78
        self.attack_pressure = 0.10
        self.previous_attack_pressure = 0.10
        self.deception_freshness = 0.50
        self.adaptation_momentum = 0.0

        self.metrics = StepMetrics()
        self.recent_rewards.clear()
        self.recent_attack_pressure.clear()
        self.recent_deception_success.clear()
        self.memory_bank.clear()

        self.attack_confidence_scores = []
        self.attack_labels = []
        self.resilience_trace = []
        self.reward_trace = []
        self.deception_trace = []
        self.compromise_trace = []
        self.action_trace = []

        if self.use_deception:
            self._initialize_decoys()

        row = self._current_flow()
        return self._compose_state(row)

    def _initialize_decoys(self) -> None:
        decoy_count = max(1, int(round(self.node_count * 0.10)))
        candidates = self.node_df.sort_values("criticality", ascending=False)["node_id"].to_numpy()
        selected = self.rng.choice(candidates, size=min(decoy_count, len(candidates)), replace=False)
        self.node_states[selected] = DECEPTIVE

    def _current_flow(self) -> pd.Series:
        index = (self.cursor + self.step_id) % len(self.dataset)
        return self.dataset.iloc[index]

    def _normalized_features(self, row: pd.Series) -> np.ndarray:
        values = row[self.feature_cols].astype(float)
        normalized = (values - self.feature_means) / (self.feature_stds + EPS)
        return normalized.to_numpy(dtype=np.float32)

    def _node_state_summary(self) -> np.ndarray:
        benign_ratio = float(np.mean(self.node_states == BENIGN))
        compromised_ratio = float(np.mean(self.node_states == COMPROMISED))
        deceptive_ratio = float(np.mean(self.node_states == DECEPTIVE))
        quarantined_ratio = float(np.mean(self.node_states == QUARANTINED))
        critical_compromised = self._critical_compromise_ratio()
        graph_density = float(nx.density(self.graph))
        average_degree = float(np.mean([degree for _, degree in self.graph.degree()]) / max(1.0, self.node_count - 1))
        return np.array(
            [
                benign_ratio,
                compromised_ratio,
                deceptive_ratio,
                quarantined_ratio,
                critical_compromised,
                graph_density,
                average_degree,
                self.deception_level,
                self.service_quality,
                self.resilience_potential,
                self.attack_pressure,
                self.deception_freshness,
                self.adaptation_momentum,
                float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0,
                float(np.mean(self.recent_attack_pressure)) if self.recent_attack_pressure else 0.0,
                float(np.mean(self.recent_deception_success)) if self.recent_deception_success else 0.0,
            ],
            dtype=np.float32,
        )

    def _compose_state(self, row: pd.Series) -> np.ndarray:
        feature_part = self._normalized_features(row)
        summary_part = self._node_state_summary()

        if self.generated_attack_samples is not None and self.use_gan and len(self.generated_attack_samples) > 0:
            gan_index = (self.step_id + self.episode_id) % len(self.generated_attack_samples)
            gan_sample = np.asarray(self.generated_attack_samples[gan_index], dtype=np.float32).flatten()
            gan_summary = np.array(
                [
                    float(np.mean(gan_sample)),
                    float(np.std(gan_sample)),
                    float(np.max(gan_sample)),
                    float(np.min(gan_sample)),
                    float(np.linalg.norm(gan_sample) / (len(gan_sample) + EPS)),
                ],
                dtype=np.float32,
            )
        else:
            gan_summary = np.zeros(5, dtype=np.float32)

        state = np.concatenate([feature_part, summary_part, gan_summary]).astype(np.float32)

        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), mode="constant")
        elif len(state) > self.state_dim:
            state = state[: self.state_dim]

        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        action = int(np.clip(action, 0, self.action_size - 1))
        action_name = self.action_names[action]
        row = self._current_flow()

        is_attack = int(row["label"]) == 1
        source_node = int(row["source_node"])
        target_node = int(row["destination_node"])
        attack_type = str(row["attack_type"])

        threat_score = self._estimate_threat_score(row)
        detection_score = self._estimate_detection_score(row, action_name, threat_score)

        action_effect = self._apply_action(action_name, source_node, target_node, threat_score, is_attack)
        compromise_success = False
        deceptive_capture = False

        if is_attack:
            self.metrics.total_attacks += 1
            compromise_success, deceptive_capture = self._propagate_attack(
                source_node=source_node,
                target_node=target_node,
                attack_type=attack_type,
                threat_score=threat_score,
                action_effect=action_effect,
            )

            if detection_score >= 0.50:
                self.metrics.true_positives += 1
            else:
                self.metrics.false_negatives += 1
        else:
            if detection_score >= 0.50:
                self.metrics.false_positives += 1
            else:
                self.metrics.true_negatives += 1

        self._update_operational_signals(
            row=row,
            action_name=action_name,
            threat_score=threat_score,
            is_attack=is_attack,
            compromise_success=compromise_success,
            deceptive_capture=deceptive_capture,
        )

        if self.use_dim:
            self._digital_immune_update()

        reward = self._compute_reward(
            is_attack=is_attack,
            compromise_success=compromise_success,
            deceptive_capture=deceptive_capture,
            action_name=action_name,
            detection_score=detection_score,
        )

        self.attack_confidence_scores.append(detection_score)
        self.attack_labels.append(int(is_attack))
        self.resilience_trace.append(self.resilience_potential)
        self.reward_trace.append(reward)
        self.deception_trace.append(self.deception_level)
        self.compromise_trace.append(int(np.sum(self.node_states == COMPROMISED)))
        self.action_trace.append(action_name)
        self.recent_rewards.append(reward)
        self.recent_attack_pressure.append(self.attack_pressure)
        self.recent_deception_success.append(1.0 if deceptive_capture else 0.0)

        self.step_id += 1
        self.done = self.step_id >= self.max_steps

        next_row = self._current_flow()
        next_state = self._compose_state(next_row)

        info = self._info_dict(
            action_name=action_name,
            is_attack=is_attack,
            attack_type=attack_type,
            threat_score=threat_score,
            detection_score=detection_score,
            compromise_success=compromise_success,
            deceptive_capture=deceptive_capture,
        )

        return next_state, float(reward), bool(self.done), info

    def _estimate_threat_score(self, row: pd.Series) -> float:
        keys = [
            "payload_anomaly_score",
            "attack_pressure",
            "protocol_entropy",
            "payload_entropy",
            "recent_alert_density",
            "lateral_scan_score",
            "dos_burst_score",
            "spoofing_consistency_gap",
            "covert_channel_score",
            "telemetry_deviation",
        ]
        values = [safe_float(row.get(key, 0.0)) for key in keys if key in row]
        base = float(np.mean(values)) if values else 0.0

        if int(row["label"]) == 1:
            base += 0.25

        if self.use_gan:
            base += 0.10 + 0.10 * float(np.mean(self.recent_attack_pressure)) if self.recent_attack_pressure else 0.10

        return float(np.clip(base, 0.0, 1.0))

    def _estimate_detection_score(self, row: pd.Series, action_name: str, threat_score: float) -> float:
        anomaly_terms = [
            safe_float(row.get("payload_anomaly_score", 0.0)),
            safe_float(row.get("protocol_entropy", 0.0)),
            safe_float(row.get("attack_pressure", 0.0)),
            safe_float(row.get("recent_alert_density", 0.0)),
            safe_float(row.get("auth_failure_rate", 0.0)),
        ]
        score = 0.55 * threat_score + 0.45 * float(np.mean(anomaly_terms))

        if action_name in {"quarantine_node", "activate_honeypot_cluster"}:
            score += 0.08
        if action_name == "maintain":
            score -= 0.04
        if self.use_deception:
            score += 0.04 * self.deception_level

        return float(np.clip(score, 0.0, 1.0))

    def _apply_action(
        self,
        action_name: str,
        source_node: int,
        target_node: int,
        threat_score: float,
        is_attack: bool,
    ) -> Dict[str, float]:
        mitigation = 0.0
        cost = 0.0
        deception_gain = 0.0
        service_penalty = 0.0

        if not self.use_deception and action_name in {
            "deploy_decoy",
            "increase_deception_level",
            "activate_honeypot_cluster",
        }:
            action_name = "maintain"

        if action_name == "maintain":
            mitigation = 0.03
            cost = 0.00

        elif action_name == "deploy_decoy":
            mitigation = 0.22
            cost = 0.06
            deception_gain = 0.20
            self._deploy_decoy_near(target_node)
            self.deception_level = float(np.clip(self.deception_level + 0.05, 0.0, 1.0))
            self.deception_freshness = 1.0

        elif action_name == "reroute_flow":
            mitigation = 0.28
            cost = 0.08
            service_penalty = 0.03
            self.metrics.reroutes += 1
            self.service_quality = float(np.clip(self.service_quality - 0.015, 0.0, 1.0))

        elif action_name == "quarantine_node":
            mitigation = 0.45
            cost = 0.12
            service_penalty = 0.05
            if threat_score > 0.45 or is_attack:
                self._quarantine_node(target_node)
            else:
                self.metrics.false_positives += 1
            self.metrics.quarantines += 1
            self.service_quality = float(np.clip(self.service_quality - 0.025, 0.0, 1.0))

        elif action_name == "increase_deception_level":
            mitigation = 0.20
            cost = 0.07
            deception_gain = 0.25
            self.deception_level = float(np.clip(self.deception_level + 0.08, 0.0, 1.0))
            self.deception_freshness = 1.0

        elif action_name == "activate_honeypot_cluster":
            mitigation = 0.52
            cost = 0.18
            deception_gain = 0.40
            service_penalty = 0.06
            self._activate_honeypot_cluster(target_node)
            self.service_quality = float(np.clip(self.service_quality - 0.02, 0.0, 1.0))
            self.deception_level = float(np.clip(self.deception_level + 0.10, 0.0, 1.0))
            self.deception_freshness = 1.0

        return {
            "mitigation": float(np.clip(mitigation, 0.0, 0.95)),
            "cost": cost,
            "deception_gain": deception_gain,
            "service_penalty": service_penalty,
        }

    def _deploy_decoy_near(self, target_node: int) -> None:
        candidates = [target_node] + list(self.graph.neighbors(target_node))
        candidates = [node for node in candidates if self.node_states[node] != QUARANTINED]
        if candidates:
            chosen = int(self.rng.choice(candidates))
            self.node_states[chosen] = DECEPTIVE

    def _activate_honeypot_cluster(self, target_node: int) -> None:
        candidates = [target_node] + list(self.graph.neighbors(target_node))
        if len(candidates) < 3:
            candidates = list(range(self.node_count))
        count = min(3, len(candidates))
        selected = self.rng.choice(candidates, size=count, replace=False)
        for node in selected:
            if self.node_states[int(node)] != QUARANTINED:
                self.node_states[int(node)] = DECEPTIVE

    def _quarantine_node(self, node: int) -> None:
        if self.node_states[node] == COMPROMISED and self.compromise_start_time[node] >= 0:
            latency = max(0.0, float(self.step_id) - float(self.compromise_start_time[node]))
            self.recovery_latencies.append(latency)
            self.metrics.recovered_nodes += 1
        self.node_states[node] = QUARANTINED
        self.compromise_start_time[node] = -1.0

    def _propagate_attack(
        self,
        source_node: int,
        target_node: int,
        attack_type: str,
        threat_score: float,
        action_effect: Dict[str, float],
    ) -> Tuple[bool, bool]:
        vulnerability = safe_float(self.graph.nodes[target_node].get("vulnerability", 0.4), 0.4)
        criticality = safe_float(self.graph.nodes[target_node].get("criticality", 0.5), 0.5)
        mitigation = action_effect["mitigation"]

        decoy_ratio = float(np.mean(self.node_states == DECEPTIVE)) if self.use_deception else 0.0
        deceptive_attraction = float(np.clip(0.20 + self.deception_level * 0.55 + decoy_ratio * 0.45, 0.0, 0.85))

        if self.node_states[target_node] == DECEPTIVE and self.use_deception:
            capture_probability = float(np.clip(0.55 + self.deception_level * 0.35 + threat_score * 0.10, 0.0, 0.98))
            if self.rng.random() < capture_probability:
                self.metrics.deceptive_captures += 1
                self.memory_bank.append({"step": self.step_id, "attack_type": attack_type, "target": int(target_node)})
                return False, True

        if self.use_deception and self.rng.random() < deceptive_attraction * 0.45:
            decoys = np.where(self.node_states == DECEPTIVE)[0]
            if len(decoys) > 0:
                self.metrics.deceptive_captures += 1
                selected_decoy = int(self.rng.choice(decoys))
                self.memory_bank.append({"step": self.step_id, "attack_type": attack_type, "target": selected_decoy})
                return False, True

        if self.node_states[target_node] == QUARANTINED:
            return False, False

        propagation_probability = (
            0.12
            + 0.42 * vulnerability
            + 0.28 * criticality
            + 0.36 * threat_score
            - 0.48 * mitigation
            - 0.32 * self.deception_level * int(self.use_deception)
        )
        propagation_probability = float(np.clip(propagation_probability, 0.01, 0.95))

        if self.rng.random() < propagation_probability:
            if self.node_states[target_node] != COMPROMISED:
                self.node_states[target_node] = COMPROMISED
                self.compromise_start_time[target_node] = float(self.step_id)
            self.metrics.successful_compromises += 1
            return True, False

        return False, False

    def _update_operational_signals(
        self,
        row: pd.Series,
        action_name: str,
        threat_score: float,
        is_attack: bool,
        compromise_success: bool,
        deceptive_capture: bool,
    ) -> None:
        self.previous_attack_pressure = self.attack_pressure

        pressure_input = threat_score if is_attack else threat_score * 0.35
        if compromise_success:
            pressure_input += 0.30
        if deceptive_capture:
            pressure_input -= 0.10

        if self.use_gan:
            pressure_input += 0.05

        self.attack_pressure = float(np.clip(0.82 * self.attack_pressure + 0.18 * pressure_input, 0.0, 1.0))

        disruption = 0.0
        if compromise_success:
            disruption += 0.09
        if action_name in {"quarantine_node", "activate_honeypot_cluster"}:
            disruption += 0.03

        recovery = 0.01
        if deceptive_capture:
            recovery += 0.025
        if action_name in {"deploy_decoy", "increase_deception_level"}:
            recovery += 0.010

        self.service_quality = float(np.clip(self.service_quality - disruption + recovery, 0.0, 1.0))
        self.deception_freshness = float(np.clip(self.deception_freshness * 0.985, 0.0, 1.0))

        if compromise_success:
            self.metrics.service_disruptions += int(self._criticality_of(target_node=int(row["destination_node"])) > 0.70)

        self._recompute_resilience()

    def _digital_immune_update(self) -> None:
        margin = float(self.dim_cfg["resilience_margin"])
        attack_w = float(self.dim_cfg["attack_pressure_weight"])
        freshness_w = float(self.dim_cfg["deception_freshness_weight"])
        service_w = float(self.dim_cfg["service_quality_weight"])

        self.resilience_potential = float(
            np.clip(
                service_w * self.service_quality
                + freshness_w * self.deception_freshness
                + attack_w * (1.0 - self.attack_pressure)
                - 0.20 * self._critical_compromise_ratio(),
                0.0,
                1.0,
            )
        )

        pressure_derivative = self.attack_pressure - self.previous_attack_pressure
        position_term = max(0.0, margin - self.resilience_potential)

        self.adaptation_momentum = float(
            np.clip(
                float(self.dim_cfg["momentum_gain"]) * position_term
                + float(self.dim_cfg["pressure_derivative_gain"]) * max(0.0, pressure_derivative),
                0.0,
                1.0,
            )
        )

        if self.resilience_potential < margin:
            self._self_heal(momentum=self.adaptation_momentum)

        interval = int(self.dim_cfg["decoy_rotation_interval"])
        if interval > 0 and self.step_id > 0 and self.step_id % interval == 0 and self.use_deception:
            self._rotate_decoys()

    def _recompute_resilience(self) -> None:
        compromise_ratio = float(np.mean(self.node_states == COMPROMISED))
        quarantine_ratio = float(np.mean(self.node_states == QUARANTINED))
        deception_support = self.deception_level * 0.20 if self.use_deception else 0.0

        self.resilience_potential = float(
            np.clip(
                0.52 * self.service_quality
                + 0.28 * (1.0 - self.attack_pressure)
                + deception_support
                - 0.28 * compromise_ratio
                - 0.08 * quarantine_ratio,
                0.0,
                1.0,
            )
        )

    def _self_heal(self, momentum: float) -> None:
        compromised = np.where(self.node_states == COMPROMISED)[0]
        if len(compromised) > 0:
            recover_count = max(1, int(np.ceil(momentum * len(compromised))))
            selected = self.rng.choice(compromised, size=min(recover_count, len(compromised)), replace=False)
            for node in selected:
                node = int(node)
                if self.compromise_start_time[node] >= 0:
                    latency = max(0.0, float(self.step_id) - float(self.compromise_start_time[node]))
                    self.recovery_latencies.append(latency)
                self.node_states[node] = BENIGN
                self.compromise_start_time[node] = -1.0
                self.metrics.recovered_nodes += 1

        if self.use_deception:
            if momentum > 0.35:
                self._rotate_decoys()
            self.deception_level = float(np.clip(self.deception_level + 0.04 * momentum, 0.0, 1.0))
            self.deception_freshness = float(np.clip(self.deception_freshness + 0.20 * momentum, 0.0, 1.0))

        self.service_quality = float(np.clip(self.service_quality + 0.08 * momentum, 0.0, 1.0))
        self._recompute_resilience()

    def _rotate_decoys(self) -> None:
        current_decoys = np.where(self.node_states == DECEPTIVE)[0]
        for node in current_decoys:
            if self.rng.random() < 0.50:
                self.node_states[int(node)] = BENIGN

        critical_nodes = self.node_df.sort_values("criticality", ascending=False)["node_id"].to_numpy()
        candidate_count = max(1, int(round(self.node_count * 0.12)))
        selected = self.rng.choice(critical_nodes, size=min(candidate_count, len(critical_nodes)), replace=False)
        for node in selected:
            node = int(node)
            if self.node_states[node] != QUARANTINED:
                self.node_states[node] = DECEPTIVE

        self.deception_freshness = 1.0

    def _compute_reward(
        self,
        is_attack: bool,
        compromise_success: bool,
        deceptive_capture: bool,
        action_name: str,
        detection_score: float,
    ) -> float:
        utility = self.service_quality
        compromise_cost = self._critical_compromise_ratio() + (0.35 if compromise_success else 0.0)
        deception_gain = (0.45 if deceptive_capture else 0.0) + 0.20 * self.deception_level
        false_alarm_penalty = 0.0

        if not is_attack and detection_score >= 0.50:
            false_alarm_penalty = 1.0

        recovery_bonus = 0.10 if self.metrics.recovered_nodes > 0 else 0.0
        service_disruption = 1.0 - self.service_quality

        action_cost = {
            "maintain": 0.00,
            "deploy_decoy": 0.04,
            "reroute_flow": 0.06,
            "quarantine_node": 0.09,
            "increase_deception_level": 0.05,
            "activate_honeypot_cluster": 0.14,
        }.get(action_name, 0.0)

        reward = (
            float(self.reward_cfg["system_utility_weight"]) * utility
            - float(self.reward_cfg["compromise_cost_weight"]) * compromise_cost
            + float(self.reward_cfg["deception_gain_weight"]) * deception_gain
            - float(self.reward_cfg["false_alarm_penalty_weight"]) * false_alarm_penalty
            + float(self.reward_cfg["recovery_bonus_weight"]) * recovery_bonus
            - float(self.reward_cfg["service_disruption_penalty_weight"]) * service_disruption
            - action_cost
        )

        return float(np.clip(reward, -2.5, 2.5))

    def _criticality_of(self, target_node: int) -> float:
        return safe_float(self.graph.nodes[target_node].get("criticality", 0.5), 0.5)

    def _critical_compromise_ratio(self) -> float:
        criticalities = np.array([self._criticality_of(node) for node in range(self.node_count)], dtype=float)
        compromised_mask = (self.node_states == COMPROMISED).astype(float)
        return float(np.sum(criticalities * compromised_mask) / (np.sum(criticalities) + EPS))

    def _info_dict(
        self,
        action_name: str,
        is_attack: bool,
        attack_type: str,
        threat_score: float,
        detection_score: float,
        compromise_success: bool,
        deceptive_capture: bool,
    ) -> Dict:
        return {
            "step": self.step_id,
            "action": action_name,
            "is_attack": bool(is_attack),
            "attack_type": attack_type,
            "threat_score": round(float(threat_score), 6),
            "detection_score": round(float(detection_score), 6),
            "compromise_success": bool(compromise_success),
            "deceptive_capture": bool(deceptive_capture),
            "compromised_nodes": int(np.sum(self.node_states == COMPROMISED)),
            "deceptive_nodes": int(np.sum(self.node_states == DECEPTIVE)),
            "quarantined_nodes": int(np.sum(self.node_states == QUARANTINED)),
            "service_quality": round(float(self.service_quality), 6),
            "attack_pressure": round(float(self.attack_pressure), 6),
            "deception_level": round(float(self.deception_level), 6),
            "resilience_potential": round(float(self.resilience_potential), 6),
            "adaptation_momentum": round(float(self.adaptation_momentum), 6),
        }

    def summarize_episode(self) -> Dict:
        total_predictions = (
            self.metrics.true_positives
            + self.metrics.true_negatives
            + self.metrics.false_positives
            + self.metrics.false_negatives
        )

        fpr = self.metrics.false_positives / max(1, self.metrics.false_positives + self.metrics.true_negatives)
        detection_rate = self.metrics.true_positives / max(1, self.metrics.true_positives + self.metrics.false_negatives)
        mttc = self._mean_time_to_compromise()
        rrl = float(np.mean(self.recovery_latencies)) if self.recovery_latencies else float(self.max_steps)
        dus = self._deception_utility_score()
        attack_surface_reduction = self._attack_surface_reduction()

        return {
            "episode": self.episode_id,
            "steps": self.step_id,
            "total_predictions": int(total_predictions),
            "total_attacks": int(self.metrics.total_attacks),
            "successful_compromises": int(self.metrics.successful_compromises),
            "deceptive_captures": int(self.metrics.deceptive_captures),
            "false_positives": int(self.metrics.false_positives),
            "true_positives": int(self.metrics.true_positives),
            "true_negatives": int(self.metrics.true_negatives),
            "false_negatives": int(self.metrics.false_negatives),
            "mttc": round(float(mttc), 6),
            "fpr": round(float(fpr), 6),
            "detection_rate": round(float(detection_rate), 6),
            "rrl": round(float(rrl), 6),
            "dus": round(float(dus), 6),
            "attack_surface_reduction": round(float(attack_surface_reduction), 6),
            "average_reward": round(float(np.mean(self.reward_trace)) if self.reward_trace else 0.0, 6),
            "final_resilience": round(float(self.resilience_potential), 6),
            "final_service_quality": round(float(self.service_quality), 6),
        }

    def _mean_time_to_compromise(self) -> float:
        if self.metrics.successful_compromises <= 0:
            return float(self.max_steps)

        compromise_steps = []
        for node in range(self.node_count):
            if self.compromise_start_time[node] >= 0:
                compromise_steps.append(float(self.compromise_start_time[node]))

        if not compromise_steps:
            return float(self.max_steps)

        return float(np.mean(compromise_steps))

    def _deception_utility_score(self) -> float:
        if self.metrics.total_attacks <= 0:
            return 0.0
        capture_component = self.metrics.deceptive_captures / max(1, self.metrics.total_attacks)
        entropy_component = entropy_from_distribution(
            [
                max(1, int(np.sum(self.node_states == BENIGN))),
                max(1, int(np.sum(self.node_states == COMPROMISED))),
                max(1, int(np.sum(self.node_states == DECEPTIVE))),
                max(1, int(np.sum(self.node_states == QUARANTINED))),
            ]
        )
        return float(np.clip(0.65 * capture_component + 0.35 * entropy_component, 0.0, 1.0))

    def _attack_surface_reduction(self) -> float:
        compromised = float(np.mean(self.node_states == COMPROMISED))
        quarantined = float(np.mean(self.node_states == QUARANTINED))
        deceptive = float(np.mean(self.node_states == DECEPTIVE))
        reduction = deceptive * 0.65 + quarantined * 0.45 - compromised * 0.35 + self.deception_level * 0.25
        return float(np.clip(reduction, 0.0, 1.0))


def heuristic_policy(env: SmartGridCyberEnv, state: np.ndarray, mode: str = "proposed") -> int:
    """
    Deterministic policy used for baselines and smoke tests.

    The learned DQN in train.py replaces this logic during training. This policy
    is still useful for reproducibility checks and baseline evaluation.
    """
    current_row = env._current_flow()
    threat = env._estimate_threat_score(current_row)

    if mode == "static_ids":
        return env.action_names.index("quarantine_node") if threat > 0.65 else env.action_names.index("maintain")

    if mode == "randomized_deception":
        if env.rng.random() < float(env.config["baselines"]["randomized_deception"]["decoy_probability"]):
            return env.action_names.index("deploy_decoy")
        return env.action_names.index("maintain")

    if mode == "non_adaptive_drl":
        if threat > 0.72:
            return env.action_names.index("quarantine_node")
        if threat > 0.55:
            return env.action_names.index("reroute_flow")
        return env.action_names.index("maintain")

    # Proposed-style safety heuristic used when a trained network is not supplied.
    if threat > 0.78:
        return env.action_names.index("activate_honeypot_cluster")
    if threat > 0.62:
        return env.action_names.index("deploy_decoy")
    if threat > 0.48:
        return env.action_names.index("increase_deception_level")
    if env.service_quality < 0.72:
        return env.action_names.index("reroute_flow")
    return env.action_names.index("maintain")


def run_policy_episode(
    env: SmartGridCyberEnv,
    policy: Optional[Callable[[SmartGridCyberEnv, np.ndarray], int]] = None,
    mode: str = "proposed",
) -> Dict:
    state = env.reset()
    done = False

    while not done:
        if policy is not None:
            action = int(policy(env, state))
        else:
            action = heuristic_policy(env, state, mode=mode)
        state, _, done, _ = env.step(action)

    return env.summarize_episode()


def save_episode_trace(env: SmartGridCyberEnv, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = pd.DataFrame(
        {
            "step": list(range(len(env.reward_trace))),
            "reward": env.reward_trace,
            "resilience": env.resilience_trace,
            "deception_level": env.deception_trace,
            "compromised_nodes": env.compromise_trace,
            "action": env.action_trace,
            "label": env.attack_labels,
            "score": env.attack_confidence_scores,
        }
    )
    trace.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for smart-grid cyber deception environment.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--mode", default="proposed", help="Policy mode for smoke test.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["reproducibility"]["global_seed"])

    env = SmartGridCyberEnv(
        config=config,
        seed=seed,
        use_gan=True,
        use_deception=True,
        use_dim=True,
        max_steps=200,
    )

    summary = run_policy_episode(env, mode=args.mode)
    result_dir = Path(config["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "environment_smoke_test_summary.json"
    trace_path = result_dir / "environment_smoke_test_trace.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_episode_trace(env, trace_path)

    print("Environment smoke test completed.")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"Trace saved to:   {trace_path}")


if __name__ == "__main__":
    main()
