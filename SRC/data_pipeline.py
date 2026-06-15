"""
Synthetic Smart-Grid SCADA-IP Dataset Generator

This module creates the reproducible dataset used by the GAN-DQN cyber deception
pipeline. It generates:
  1. Smart-grid communication topology
  2. Node role metadata
  3. Benign SCADA-IP/DNP3-style traffic
  4. Adversarial traffic for multiple attack families
  5. Train/validation/test CSV files
  6. Dataset metadata and statistical summary reports

The generated data are synthetic and are created locally. CICIDS2017 and
UNSW-NB15 are referenced in the README for external benchmark alignment only;
they are not redistributed or required by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml


EPS = 1e-9


@dataclass(frozen=True)
class DatasetPaths:
    generated_dir: Path
    topology_nodes: Path
    topology_edges: Path
    full_dataset: Path
    train_dataset: Path
    validation_dataset: Path
    test_dataset: Path
    metadata: Path
    feature_schema: Path
    statistical_report: Path


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def make_paths(config: Dict) -> DatasetPaths:
    generated_dir = Path(config["paths"]["generated_data_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)

    return DatasetPaths(
        generated_dir=generated_dir,
        topology_nodes=generated_dir / "smartgrid_nodes.csv",
        topology_edges=generated_dir / "smartgrid_edges.csv",
        full_dataset=generated_dir / "smartgrid_scada_ip_dataset.csv",
        train_dataset=generated_dir / "train.csv",
        validation_dataset=generated_dir / "validation.csv",
        test_dataset=generated_dir / "test.csv",
        metadata=generated_dir / "dataset_metadata.json",
        feature_schema=generated_dir / "feature_schema.json",
        statistical_report=generated_dir / "statistical_alignment_report.json",
    )


def build_smartgrid_topology(config: Dict, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, nx.Graph]:
    nodes = int(config["smartgrid"]["nodes"])
    links = int(config["smartgrid"]["links"])
    zones = int(config["smartgrid"]["regional_control_zones"])

    graph = nx.Graph()
    graph.add_nodes_from(range(nodes))

    # Ring backbone keeps the grid connected.
    for idx in range(nodes):
        graph.add_edge(idx, (idx + 1) % nodes)

    # Add zone-level and cross-zone links until the requested edge count is reached.
    while graph.number_of_edges() < links:
        u = int(rng.integers(0, nodes))
        v = int(rng.integers(0, nodes))
        if u != v:
            graph.add_edge(u, v)

    role_pool = []
    role_pool.extend(["SCADA_MASTER"])
    role_pool.extend(["REGIONAL_CONTROLLER"] * zones)
    role_pool.extend(["RTU"] * 10)
    role_pool.extend(["IED"] * 10)
    role_pool.extend(["PMU"] * 5)
    role_pool.extend(["SENSOR"] * max(0, nodes - len(role_pool)))
    role_pool = role_pool[:nodes]

    rng.shuffle(role_pool)

    node_rows = []
    for node_id in range(nodes):
        role = role_pool[node_id]
        zone = 0 if role == "SCADA_MASTER" else int((node_id % zones) + 1)
        criticality = {
            "SCADA_MASTER": 1.00,
            "REGIONAL_CONTROLLER": 0.88,
            "RTU": 0.72,
            "IED": 0.64,
            "PMU": 0.58,
            "SENSOR": 0.42,
        }[role]
        vulnerability = float(np.clip(rng.normal(0.42 + (1.0 - criticality) * 0.25, 0.12), 0.05, 0.95))
        node_rows.append(
            {
                "node_id": node_id,
                "role": role,
                "zone": zone,
                "criticality": round(criticality, 4),
                "baseline_vulnerability": round(vulnerability, 4),
                "initial_state": "benign",
            }
        )

    edge_rows = []
    for u, v in graph.edges():
        bandwidth = float(np.clip(rng.normal(config["smartgrid"]["bandwidth_mbps"], 12.0), 40.0, 140.0))
        latency = float(
            np.clip(
                rng.normal(config["smartgrid"]["base_latency_ms"], config["smartgrid"]["latency_jitter_ms"]),
                2.0,
                70.0,
            )
        )
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "bandwidth_mbps": round(bandwidth, 4),
                "latency_ms": round(latency, 4),
                "link_type": "control" if rng.random() < 0.35 else "telemetry",
            }
        )

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), graph


def base_feature_names(feature_count: int) -> List[str]:
    core_features = [
        "flow_duration_ms",
        "fwd_packet_count",
        "bwd_packet_count",
        "total_fwd_bytes",
        "total_bwd_bytes",
        "packet_size_mean",
        "packet_size_std",
        "packet_size_max",
        "interarrival_mean_ms",
        "interarrival_std_ms",
        "flow_bytes_per_sec",
        "flow_packets_per_sec",
        "protocol_entropy",
        "ttl_mean",
        "ttl_std",
        "fragmentation_ratio",
        "tcp_syn_ratio",
        "tcp_rst_ratio",
        "tcp_ack_ratio",
        "dnp3_function_entropy",
        "scada_command_rate",
        "telemetry_deviation",
        "setpoint_change_rate",
        "auth_failure_rate",
        "source_destination_correlation",
        "flow_persistence",
        "connection_attempt_rate",
        "payload_entropy",
        "payload_anomaly_score",
        "port_switching_rate",
        "failed_command_ratio",
        "replay_similarity",
        "lateral_scan_score",
        "dos_burst_score",
        "spoofing_consistency_gap",
        "covert_channel_score",
        "device_role_risk",
        "zone_crossing_ratio",
        "critical_path_proximity",
        "recent_alert_density",
        "deception_entropy",
        "attacker_belief_shift",
        "service_quality_index",
        "resilience_potential",
        "attack_pressure",
        "decoy_exposure_score",
        "recovery_readiness",
        "routing_volatility",
    ]

    if feature_count <= len(core_features):
        return core_features[:feature_count]

    extra = [f"latent_behavior_feature_{i:02d}" for i in range(1, feature_count - len(core_features) + 1)]
    return core_features + extra


def choose_attack_type(attack_types: List[str], rng: np.random.Generator) -> str:
    weights = np.array([0.18, 0.14, 0.16, 0.17, 0.16, 0.10, 0.09], dtype=float)
    if len(attack_types) != len(weights):
        weights = np.ones(len(attack_types), dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(attack_types, p=weights))


def attack_signature(attack_type: str) -> Dict[str, float]:
    signatures = {
        "command_injection": {
            "scada_command_rate": 2.8,
            "failed_command_ratio": 2.6,
            "payload_anomaly_score": 2.1,
            "setpoint_change_rate": 2.0,
        },
        "protocol_fuzzing": {
            "protocol_entropy": 2.4,
            "dnp3_function_entropy": 2.6,
            "fragmentation_ratio": 1.8,
            "payload_entropy": 2.2,
        },
        "lateral_movement": {
            "lateral_scan_score": 2.8,
            "connection_attempt_rate": 2.2,
            "zone_crossing_ratio": 2.1,
            "port_switching_rate": 1.9,
        },
        "coordinated_dos": {
            "dos_burst_score": 3.2,
            "flow_packets_per_sec": 2.8,
            "flow_bytes_per_sec": 2.4,
            "interarrival_mean_ms": -1.6,
        },
        "spoofed_telemetry": {
            "spoofing_consistency_gap": 2.9,
            "telemetry_deviation": 2.5,
            "source_destination_correlation": -1.4,
            "replay_similarity": 1.4,
        },
        "replay_attack": {
            "replay_similarity": 3.0,
            "flow_persistence": 1.7,
            "telemetry_deviation": 1.5,
            "attacker_belief_shift": 1.2,
        },
        "covert_channel_flooding": {
            "covert_channel_score": 3.1,
            "payload_entropy": 2.3,
            "packet_size_std": 1.6,
            "flow_persistence": 1.5,
        },
    }
    return signatures.get(attack_type, {})


def generate_single_flow(
    feature_names: List[str],
    node_df: pd.DataFrame,
    graph: nx.Graph,
    attack_types: List[str],
    benign_ratio: float,
    rng: np.random.Generator,
    timestamp_id: int,
) -> Dict[str, object]:
    source = int(rng.integers(0, len(node_df)))
    neighbor_list = list(graph.neighbors(source))
    destination = int(rng.choice(neighbor_list)) if neighbor_list else int(rng.integers(0, len(node_df)))

    src_row = node_df.loc[node_df["node_id"] == source].iloc[0]
    dst_row = node_df.loc[node_df["node_id"] == destination].iloc[0]

    is_attack = rng.random() > benign_ratio
    attack_type = "benign" if not is_attack else choose_attack_type(attack_types, rng)

    criticality = max(float(src_row["criticality"]), float(dst_row["criticality"]))
    vulnerability = max(float(src_row["baseline_vulnerability"]), float(dst_row["baseline_vulnerability"]))
    cross_zone = 1.0 if int(src_row["zone"]) != int(dst_row["zone"]) else 0.0

    # Core benign operating profile.
    values = {
        "flow_duration_ms": rng.gamma(2.2, 38.0),
        "fwd_packet_count": rng.poisson(9) + 1,
        "bwd_packet_count": rng.poisson(7) + 1,
        "total_fwd_bytes": rng.gamma(3.0, 280.0),
        "total_bwd_bytes": rng.gamma(2.8, 240.0),
        "packet_size_mean": rng.normal(180.0, 28.0),
        "packet_size_std": rng.normal(42.0, 8.0),
        "packet_size_max": rng.normal(520.0, 75.0),
        "interarrival_mean_ms": rng.normal(18.0, 4.0),
        "interarrival_std_ms": rng.normal(6.5, 2.0),
        "flow_bytes_per_sec": rng.gamma(2.5, 950.0),
        "flow_packets_per_sec": rng.gamma(2.2, 34.0),
        "protocol_entropy": rng.normal(0.32, 0.08),
        "ttl_mean": rng.normal(61.0, 3.5),
        "ttl_std": rng.normal(1.8, 0.45),
        "fragmentation_ratio": rng.beta(1.2, 18.0),
        "tcp_syn_ratio": rng.beta(2.0, 10.0),
        "tcp_rst_ratio": rng.beta(1.2, 22.0),
        "tcp_ack_ratio": rng.beta(9.0, 3.0),
        "dnp3_function_entropy": rng.normal(0.28, 0.07),
        "scada_command_rate": rng.beta(1.4, 12.0),
        "telemetry_deviation": abs(rng.normal(0.08, 0.035)),
        "setpoint_change_rate": rng.beta(1.2, 16.0),
        "auth_failure_rate": rng.beta(1.0, 30.0),
        "source_destination_correlation": rng.normal(0.84, 0.07),
        "flow_persistence": rng.beta(4.5, 3.0),
        "connection_attempt_rate": rng.gamma(1.2, 1.5),
        "payload_entropy": rng.normal(0.42, 0.09),
        "payload_anomaly_score": rng.beta(1.2, 14.0),
        "port_switching_rate": rng.beta(1.1, 18.0),
        "failed_command_ratio": rng.beta(1.1, 25.0),
        "replay_similarity": rng.beta(1.4, 9.0),
        "lateral_scan_score": rng.beta(1.0, 18.0),
        "dos_burst_score": rng.beta(1.0, 20.0),
        "spoofing_consistency_gap": rng.beta(1.2, 17.0),
        "covert_channel_score": rng.beta(1.0, 22.0),
        "device_role_risk": criticality,
        "zone_crossing_ratio": cross_zone,
        "critical_path_proximity": criticality * (0.55 + 0.35 * rng.random()),
        "recent_alert_density": rng.beta(1.3, 16.0),
        "deception_entropy": rng.beta(2.0, 4.0),
        "attacker_belief_shift": rng.beta(1.5, 8.0),
        "service_quality_index": rng.normal(0.92, 0.04),
        "resilience_potential": rng.normal(0.78, 0.07),
        "attack_pressure": rng.beta(1.1, 12.0),
        "decoy_exposure_score": rng.beta(1.5, 7.0),
        "recovery_readiness": rng.normal(0.74, 0.08),
        "routing_volatility": rng.beta(1.2, 11.0),
    }

    if is_attack:
        severity = float(np.clip(rng.normal(1.0 + vulnerability * 0.7 + criticality * 0.4, 0.18), 0.75, 2.25))

        generic_attack_shift = {
            "protocol_entropy": 0.35,
            "payload_entropy": 0.38,
            "payload_anomaly_score": 0.42,
            "auth_failure_rate": 0.22,
            "recent_alert_density": 0.38,
            "attack_pressure": 0.55,
            "service_quality_index": -0.18,
            "resilience_potential": -0.20,
            "attacker_belief_shift": 0.32,
            "routing_volatility": 0.30,
        }

        for key, shift in generic_attack_shift.items():
            values[key] = values[key] + shift * severity

        for key, strength in attack_signature(attack_type).items():
            if key in values:
                values[key] = values[key] + strength * severity * 0.22

        # Attack-specific volumetric shifts.
        if attack_type == "coordinated_dos":
            values["fwd_packet_count"] *= rng.uniform(3.0, 7.5)
            values["bwd_packet_count"] *= rng.uniform(1.5, 4.0)
            values["total_fwd_bytes"] *= rng.uniform(2.5, 6.5)
            values["interarrival_mean_ms"] *= rng.uniform(0.10, 0.38)

        if attack_type in {"lateral_movement", "protocol_fuzzing"}:
            destination = int(rng.integers(0, len(node_df)))
            values["zone_crossing_ratio"] = 1.0

        if attack_type == "spoofed_telemetry":
            values["source_destination_correlation"] *= rng.uniform(0.30, 0.72)
            values["telemetry_deviation"] += rng.uniform(0.35, 0.75)

    # Derived and latent features.
    for feature in feature_names:
        if feature.startswith("latent_behavior_feature_"):
            base = rng.normal(0.0, 1.0)
            if is_attack:
                base += rng.normal(1.35, 0.40)
            values[feature] = base

    clipped_values = {}
    for feature in feature_names:
        value = float(values.get(feature, rng.normal(0.0, 1.0)))
        if feature in {
            "protocol_entropy",
            "fragmentation_ratio",
            "tcp_syn_ratio",
            "tcp_rst_ratio",
            "tcp_ack_ratio",
            "dnp3_function_entropy",
            "scada_command_rate",
            "telemetry_deviation",
            "setpoint_change_rate",
            "auth_failure_rate",
            "source_destination_correlation",
            "flow_persistence",
            "payload_entropy",
            "payload_anomaly_score",
            "port_switching_rate",
            "failed_command_ratio",
            "replay_similarity",
            "lateral_scan_score",
            "dos_burst_score",
            "spoofing_consistency_gap",
            "covert_channel_score",
            "device_role_risk",
            "zone_crossing_ratio",
            "critical_path_proximity",
            "recent_alert_density",
            "deception_entropy",
            "attacker_belief_shift",
            "service_quality_index",
            "resilience_potential",
            "attack_pressure",
            "decoy_exposure_score",
            "recovery_readiness",
            "routing_volatility",
        }:
            value = float(np.clip(value, 0.0, 1.0))
        else:
            value = max(value, 0.0)
        clipped_values[feature] = round(value, 6)

    row = {
        "timestamp_id": timestamp_id,
        "source_node": source,
        "destination_node": destination,
        "source_role": str(src_row["role"]),
        "destination_role": str(dst_row["role"]),
        "source_zone": int(src_row["zone"]),
        "destination_zone": int(dst_row["zone"]),
        "protocol": "DNP3_STYLE" if rng.random() < 0.62 else "TCP_IP",
        "attack_type": attack_type,
        "label": int(is_attack),
    }
    row.update(clipped_values)
    return row


def generate_dataset(config: Dict, node_df: pd.DataFrame, graph: nx.Graph, rng: np.random.Generator) -> pd.DataFrame:
    feature_count = int(config["smartgrid"]["feature_count"])
    feature_names = base_feature_names(feature_count)
    attack_types = list(config["traffic"]["attack_types"])
    benign_ratio = float(config["traffic"]["benign_ratio"])
    samples = int(config["traffic"]["samples_per_run"])

    rows = [
        generate_single_flow(
            feature_names=feature_names,
            node_df=node_df,
            graph=graph,
            attack_types=attack_types,
            benign_ratio=benign_ratio,
            rng=rng,
            timestamp_id=idx,
        )
        for idx in range(samples)
    ]

    return pd.DataFrame(rows)


def split_dataset(dataset: pd.DataFrame, config: Dict, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled_idx = np.arange(len(dataset))
    rng.shuffle(shuffled_idx)
    dataset = dataset.iloc[shuffled_idx].reset_index(drop=True)

    train_ratio = float(config["traffic"]["train_split"])
    val_ratio = float(config["traffic"]["validation_split"])

    train_end = int(len(dataset) * train_ratio)
    val_end = train_end + int(len(dataset) * val_ratio)

    train = dataset.iloc[:train_end].reset_index(drop=True)
    validation = dataset.iloc[train_end:val_end].reset_index(drop=True)
    test = dataset.iloc[val_end:].reset_index(drop=True)

    return train, validation, test


def feature_columns(dataset: pd.DataFrame) -> List[str]:
    metadata_columns = {
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
    return [col for col in dataset.columns if col not in metadata_columns]


def safe_histogram(values: np.ndarray, bins: int = 30) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, density=False)
    hist = hist.astype(float) + EPS
    return hist / hist.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p + EPS
    q = q + EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def statistical_report(dataset: pd.DataFrame, features: List[str]) -> Dict:
    benign = dataset[dataset["label"] == 0]
    attack = dataset[dataset["label"] == 1]

    key_features = [
        "protocol_entropy",
        "payload_entropy",
        "payload_anomaly_score",
        "connection_attempt_rate",
        "telemetry_deviation",
        "attack_pressure",
        "service_quality_index",
        "resilience_potential",
    ]
    key_features = [feature for feature in key_features if feature in features]

    feature_kl = {}
    for feature in key_features:
        combined_min = float(dataset[feature].min())
        combined_max = float(dataset[feature].max())
        if math.isclose(combined_min, combined_max):
            feature_kl[feature] = 0.0
            continue

        benign_hist, bin_edges = np.histogram(benign[feature].to_numpy(), bins=30, range=(combined_min, combined_max))
        attack_hist, _ = np.histogram(attack[feature].to_numpy(), bins=30, range=(combined_min, combined_max))
        benign_hist = benign_hist.astype(float) + EPS
        attack_hist = attack_hist.astype(float) + EPS
        feature_kl[feature] = round(kl_divergence(attack_hist, benign_hist), 6)

    feature_corr = dataset[features].corr(numeric_only=True).fillna(0.0)
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    average_abs_correlation = float(upper.abs().stack().mean()) if not upper.abs().stack().empty else 0.0

    attack_distribution = dataset["attack_type"].value_counts(normalize=True).to_dict()
    label_distribution = dataset["label"].value_counts(normalize=True).to_dict()

    return {
        "dataset_type": "synthetic_smartgrid_scada_ip",
        "total_samples": int(len(dataset)),
        "feature_count": int(len(features)),
        "label_distribution": {str(k): round(float(v), 6) for k, v in label_distribution.items()},
        "attack_type_distribution": {str(k): round(float(v), 6) for k, v in attack_distribution.items()},
        "key_feature_kl_divergence_attack_vs_benign": feature_kl,
        "average_absolute_feature_correlation": round(average_abs_correlation, 6),
        "benchmark_alignment_note": (
            "The generated dataset uses flow-level, protocol-level, temporal, and anomaly-related "
            "features commonly used in intrusion detection benchmarks. CICIDS2017 and UNSW-NB15 "
            "are not required by the code and are referenced only as external public datasets for "
            "future validation and benchmark alignment."
        ),
    }


def write_feature_schema(path: Path, features: List[str]) -> None:
    schema = {
        "feature_count": len(features),
        "feature_columns": features,
        "target_column": "label",
        "attack_column": "attack_type",
        "metadata_columns": [
            "timestamp_id",
            "source_node",
            "destination_node",
            "source_role",
            "destination_role",
            "source_zone",
            "destination_zone",
            "protocol",
        ],
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def write_metadata(path: Path, config: Dict, dataset: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    metadata = {
        "project": config["project"]["name"],
        "version": config["project"]["version"],
        "dataset_name": "Synthetic Smart-Grid SCADA-IP Cyber Deception Dataset",
        "dataset_generation": "local_protocol_aware_synthetic_simulation",
        "total_samples": int(len(dataset)),
        "train_samples": int(len(train)),
        "validation_samples": int(len(validation)),
        "test_samples": int(len(test)),
        "nodes": int(config["smartgrid"]["nodes"]),
        "links": int(config["smartgrid"]["links"]),
        "regional_control_zones": int(config["smartgrid"]["regional_control_zones"]),
        "central_scada_master": int(config["smartgrid"]["central_scada_master"]),
        "protocols": list(config["smartgrid"]["protocols"]),
        "attack_types": list(config["traffic"]["attack_types"]),
        "public_benchmark_references": {
            "CICIDS2017": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "UNSW-NB15": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        },
        "redistribution_note": (
            "The primary dataset is synthetic and generated by this repository. "
            "External public datasets are not redistributed."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic smart-grid SCADA-IP dataset.")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file.")
    args = parser.parse_args()

    config = load_config(args.config)
    rng = set_global_seed(int(config["reproducibility"]["global_seed"]))
    paths = make_paths(config)

    print("Generating smart-grid topology...")
    node_df, edge_df, graph = build_smartgrid_topology(config, rng)
    node_df.to_csv(paths.topology_nodes, index=False)
    edge_df.to_csv(paths.topology_edges, index=False)

    print("Generating synthetic SCADA-IP traffic...")
    dataset = generate_dataset(config, node_df, graph, rng)
    features = feature_columns(dataset)

    train, validation, test = split_dataset(dataset, config, rng)

    dataset.to_csv(paths.full_dataset, index=False)
    train.to_csv(paths.train_dataset, index=False)
    validation.to_csv(paths.validation_dataset, index=False)
    test.to_csv(paths.test_dataset, index=False)

    write_metadata(paths.metadata, config, dataset, train, validation, test)
    write_feature_schema(paths.feature_schema, features)

    report = statistical_report(dataset, features)
    paths.statistical_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Dataset generation completed.")
    print(f"  Full dataset: {paths.full_dataset}")
    print(f"  Train set:    {paths.train_dataset}")
    print(f"  Validation:   {paths.validation_dataset}")
    print(f"  Test set:     {paths.test_dataset}")
    print(f"  Metadata:     {paths.metadata}")
    print(f"  Report:       {paths.statistical_report}")


if __name__ == "__main__":
    main()
