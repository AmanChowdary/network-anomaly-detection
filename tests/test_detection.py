"""Tests for network anomaly detection pipeline."""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.generate_data import generate_network_data
from features.extract import extract_features, prepare_features

@pytest.fixture(scope="module")
def sample_df():
    return generate_network_data(n=5000, seed=0)

def test_data_generation(sample_df):
    assert len(sample_df) == 5000
    assert "label" in sample_df.columns
    assert sample_df["label"].isin([0, 1]).all()
    anomaly_rate = sample_df["label"].mean()
    assert 0.01 <= anomaly_rate <= 0.20

def test_feature_extraction(sample_df):
    feat = extract_features(sample_df)
    assert "bytes_ratio" in feat.columns
    assert "log_src_bytes" in feat.columns
    assert "syn_only" in feat.columns
    assert "proto_tcp" in feat.columns
    assert feat.isnull().sum().sum() == 0

def test_prepare_features(sample_df):
    X, y = prepare_features(sample_df)
    assert len(X) == len(sample_df)
    assert len(y) == len(sample_df)
    assert X.isnull().sum().sum() == 0
    assert X.shape[1] > 20

def test_model_training():
    from model.train import train
    os.makedirs("data", exist_ok=True)
    os.makedirs("model/artifacts", exist_ok=True)
    df = generate_network_data(n=5000, seed=42)
    df.to_csv("data/network_traffic.csv", index=False)
    model, metrics = train("data/network_traffic.csv", "model/artifacts")
    assert metrics["f1"] >= 0.50
    assert metrics["auc_roc"] >= 0.80

def test_alert_manager():
    from alerts.alerting import AlertManager
    manager = AlertManager(alert_log="alerts/test_alerts.jsonl")
    event = {"flow_id": "F1", "src_ip": "10.0.0.1", "attack_type": "dos",
             "src_bytes": 1e8, "dst_bytes": 100}
    # High probability → should fire
    alert = manager.fire(event, prob=0.92, predicted_class=1)
    assert alert is not None
    assert alert["severity"] == "HIGH"
    # Duplicate within window → should suppress
    alert2 = manager.fire(event, prob=0.92, predicted_class=1)
    assert alert2 is None
