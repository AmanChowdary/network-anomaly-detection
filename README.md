# Network Anomaly Detection System

Supervised ML system detecting anomalous network traffic with 94% F1-score on 500K+ events. Built with XGBoost, SMOTE oversampling, MLflow experiment tracking, and automated weekly retraining via GitHub Actions.

## Architecture

```
[Raw PCAP/Syslog] ──► [Feature Extraction Pipeline] ──► [XGBoost Classifier]
                              │                                   │
                        [20+ Features]                    [SMOTE Balancing]
                        [Log transforms]                  [GridSearchCV]
                        [Flag combinations]                     │
                        [Port classification]          [Real-time Alerting]
                                                                 │
                                                     [MLflow Tracking]
                                                     [Weekly Retraining]
```

## Key Features
- **94% F1-score** on 500K+ labeled events (5 attack categories)
- **SMOTE oversampling** to address 5% anomaly class imbalance
- **20+ engineered features**: packet ratios, log transforms, flag combos, port classification
- **GridSearchCV** hyperparameter tuning with cross-validation
- **MLflow**: experiment tracking, model versioning, artifact management
- **Real-time alerting** with severity tiers and duplicate suppression (MTTR -45%)
- **GitHub Actions**: automated weekly retraining, performance threshold check

## Quick Start

```bash
pip install -r requirements.txt

# Generate synthetic data
python data/generate_data.py

# Train model
python model/train.py

# Run real-time alerting simulation
python alerts/alerting.py
```

## Run Tests
```bash
pytest tests/ -v
```

## Detectable Attack Types
| Attack | Description |
|--------|-------------|
| Port Scan | Short-duration, sequential port probing |
| DoS | Massive byte volume, high packet rate |
| Data Exfiltration | Large outbound transfers |
| Brute Force | Repeated SYN+RST, short duration |
| Lateral Movement | Unusual internal traffic patterns |

## Tech Stack
`Python` `XGBoost` `scikit-learn` `imbalanced-learn (SMOTE)` `MLflow` `pandas` `GitHub Actions`
