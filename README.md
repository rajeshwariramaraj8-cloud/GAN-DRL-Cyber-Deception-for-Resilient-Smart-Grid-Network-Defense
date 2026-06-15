# GAN-DRL Cyber Deception for Resilient Smart-Grid Network Defense

This repository provides the reproducible implementation of a self-adaptive cyber deception framework for resilient smart-grid network defense. The framework combines adversarial traffic generation, reinforcement-based deception control, and digital immune-inspired recovery to evaluate proactive defense under evolving attack conditions.

The implementation supports synthetic smart-grid SCADA-IP traffic generation, GAN-based adversarial sample synthesis, DQN-based adaptive defense learning, baseline comparison, ablation analysis, statistical evaluation, and result visualization.

## Repository Status

This repository is prepared for archival in a DOI-minting repository such as Zenodo or Code Ocean.

DOI: To be added after repository archival.

## Core Method

The project implements the following components:

- Synthetic smart-grid SCADA-IP network simulation
- Protocol-aware benign and adversarial traffic generation
- GAN-based adversarial attack sample generation
- DQN-based adaptive cyber deception policy learning
- Digital Immune Mechanism for self-healing and resilience adaptation
- Baseline comparison against static IDS, anomaly detection, non-adaptive DRL, and randomized deception
- Evaluation using MTTC, FPR, RRL, DUS, AUC, attack surface reduction, and statistical significance testing

## Repository Structure

```text
GAN-DRL-Cyber-Deception-SmartGrid/
│
├── README.md
├── requirements.txt
├── config.yaml
├── run_reproducibility.sh
├── .zenodo.json
│
└── src/
    ├── data_pipeline.py
    ├── smartgrid_env.py
    ├── models.py
    ├── train.py
    ├── evaluate.py
    └── plots.py
    
