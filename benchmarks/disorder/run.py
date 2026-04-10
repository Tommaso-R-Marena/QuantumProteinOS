import argparse
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
try:
    from qpos.disorder.ensemble import DisorderNetV6
    from qpos.disorder.disprot_loader import download_disprot
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Use 500-sequence subset')
    parser.add_argument('--full', action='store_true', help='Full evaluation')
    args = parser.parse_args()

    print(f"Running DisorderNet benchmark {'(fast)' if args.fast else '(full)'}...")
    
    df = download_disprot(version="CAID3")
    
    if args.fast:
        if len(df) > 500:
            df = df.sample(500, random_state=42).reset_index(drop=True)
        train_df = df.iloc[:400]
        test_df = df.iloc[400:]
    else:
        train_len = int(len(df) * 0.8)
        train_df = df.iloc[:train_len]
        test_df = df.iloc[train_len:]
        
    model = DisorderNetV6()
    print(f"Training on {len(train_df)} sequences...")
    model.fit(train_df['sequence'].values, train_df['disorder_labels'].values)
    
    all_y_true = []
    all_y_pred = []
    
    print(f"Evaluating on {len(test_df)} sequences...")
    for _, row in test_df.iterrows():
        seq = row['sequence']
        labels = row['disorder_labels']
        if isinstance(labels, str):
            labels = eval(labels)
        preds = model.predict(seq)
        all_y_true.extend(labels)
        all_y_pred.extend(preds)
        
    if len(np.unique(all_y_true)) > 1:
        auc = roc_auc_score(all_y_true, all_y_pred)
    else:
        auc = 0.5 
        
    print(f"AUC: {auc:.3f}")
    assert auc >= 0.70, f'DisorderNet AUC {auc:.3f} below threshold 0.70 (fast mode)'
        
    os.makedirs('benchmarks/disorder/results', exist_ok=True)
    with open('benchmarks/disorder/results/metrics.json', 'w') as f:
        json.dump({'auc_roc': auc}, f)
        
if __name__ == '__main__':
    main()
