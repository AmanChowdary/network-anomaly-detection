"""
Generates synthetic network traffic dataset (mimics PCAP/syslog feature vectors).
500K events with ~5% anomaly rate (class imbalance addressed via SMOTE in training).
"""
import pandas as pd
import numpy as np
import os

def generate_network_data(n=500000, anomaly_rate=0.05, seed=42):
    np.random.seed(seed)
    n_normal = int(n * (1 - anomaly_rate))
    n_anomaly = n - n_normal

    def normal_traffic(size):
        return {
            "duration_sec":    np.random.exponential(2, size).clip(0.001, 3600),
            "src_bytes":       np.random.lognormal(8, 2, size).clip(0, 1e8),
            "dst_bytes":       np.random.lognormal(7, 2, size).clip(0, 1e8),
            "src_packets":     np.random.poisson(20, size).clip(1, 10000),
            "dst_packets":     np.random.poisson(18, size).clip(0, 10000),
            "packet_size_avg": np.random.normal(500, 150, size).clip(40, 9000),
            "packet_size_std": np.random.exponential(100, size).clip(0, 3000),
            "inter_arrival_ms":np.random.exponential(50, size).clip(0.1, 60000),
            "protocol":        np.random.choice([6, 17, 1, 132], size, p=[0.7, 0.2, 0.08, 0.02]),
            "src_port":        np.random.choice(
                                   [80,443,8080,22,3306,5432,6379,27017], size),
            "dst_port":        np.random.randint(1024, 65535, size),
            "flags_syn":       np.random.binomial(1, 0.3, size),
            "flags_ack":       np.random.binomial(1, 0.85, size),
            "flags_fin":       np.random.binomial(1, 0.25, size),
            "flags_rst":       np.random.binomial(1, 0.02, size),
            "land":            np.zeros(size, dtype=int),
            "wrong_fragment":  np.random.binomial(1, 0.001, size),
            "urgent":          np.zeros(size, dtype=int),
            "protocol_entropy":np.random.normal(0.6, 0.1, size).clip(0, 1),
            "label":           np.zeros(size, dtype=int),
        }

    def anomaly_traffic(size):
        attack_type = np.random.choice(
            ["port_scan","dos","data_exfil","brute_force","lateral_movement"],
            size, p=[0.3, 0.3, 0.2, 0.1, 0.1])
        data = normal_traffic(size)
        # Port scan: many short connections
        mask = attack_type == "port_scan"
        data["duration_sec"][mask] = np.random.uniform(0.001, 0.1, mask.sum())
        data["src_packets"][mask] = np.random.randint(1, 5, mask.sum())
        data["dst_port"][mask] = np.random.randint(1, 1024, mask.sum())
        # DoS: huge byte volume
        mask = attack_type == "dos"
        data["src_bytes"][mask] = np.random.uniform(1e7, 1e9, mask.sum())
        data["src_packets"][mask] = np.random.randint(10000, 100000, mask.sum())
        # Data exfil: large outbound
        mask = attack_type == "data_exfil"
        data["dst_bytes"][mask] = np.random.uniform(5e6, 5e8, mask.sum())
        # Brute force: repeated SYN
        mask = attack_type == "brute_force"
        data["flags_syn"][mask] = 1
        data["flags_rst"][mask] = 1
        data["duration_sec"][mask] = np.random.uniform(0.01, 0.5, mask.sum())
        data["label"] = np.ones(size, dtype=int)
        data["attack_type"] = attack_type
        return data

    print(f"Generating {n_normal:,} normal + {n_anomaly:,} anomaly events...")
    norm = pd.DataFrame(normal_traffic(n_normal))
    norm["attack_type"] = "normal"
    anom = pd.DataFrame(anomaly_traffic(n_anomaly))

    df = pd.concat([norm, anom], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["flow_id"] = [f"FLOW_{i:09d}" for i in range(len(df))]

    print(f"Dataset: {len(df):,} events | anomaly rate: {df['label'].mean():.2%}")
    print(f"Attack distribution:\n{df['attack_type'].value_counts().to_string()}")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_network_data(n=100000)
    df.to_csv("data/network_traffic.csv", index=False)
    print("Saved → data/network_traffic.csv")
