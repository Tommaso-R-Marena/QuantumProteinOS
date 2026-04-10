import argparse
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
try:
    from qpos.disorder.ensemble import DisorderNetV6
    from qpos.disorder.disprot_loader import download_disprot
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Use 500-sequence subset (smoke test)')
    parser.add_argument('--full', action='store_true', help='Full evaluation on entire DisProt CAID3')
    args = parser.parse_args()

    print(f"Running DisorderNet benchmark {'(fast)' if args.fast else '(full)'}...")
    
    df = download_disprot(version="CAID3")
    
    # Process string labels before statistics
    def parse_labels(lbl):
        if isinstance(lbl, str):
            import ast
            try:
                return ast.literal_eval(lbl)
            except Exception:
                return [int(x) for x in lbl.strip('[]').split(',')]
        return lbl
        
    df['disorder_labels'] = df['disorder_labels'].apply(parse_labels)
    
    # Stratified splitting
    df['pct_disordered'] = df['disorder_labels'].apply(lambda x: np.mean(x) if len(x)>0 else 0)
    df['is_mostly_disordered'] = (df['pct_disordered'] > 0.3).astype(int)
    
    if args.fast:
        # Smoke test: 350 train / 150 test from a 500-seq stratified subset
        # Purpose: confirm pipeline runs and AUC >= 0.65. NOT the paper number.
        if len(df) > 500:
            df, _ = train_test_split(df, train_size=500, random_state=42, stratify=df['is_mostly_disordered'])
        train_df, test_df = train_test_split(
            df, test_size=150, random_state=42,
            stratify=df['is_mostly_disordered']
        )
        threshold = 0.65
        mode_str = "fast mode (smoke test)"
    else:
        # Full mode: 80/20 stratified split on entire DisProt CAID3
        # This is the number that goes in the paper.
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df['is_mostly_disordered']
        )
        threshold = 0.82
        mode_str = "full mode"
        
    # Check 1: Data leakage
    train_ids = set(train_df['uniprot_id'].tolist())
    test_ids = set(test_df['uniprot_id'].tolist())
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} sequences in both train and test"

    # Check 2: Label distribution sanity check
    all_train_labels = np.concatenate(train_df['disorder_labels'].tolist())
    disorder_fraction = all_train_labels.mean()
    print(f"Training label distribution: {disorder_fraction:.2%} disordered")
    assert 0.05 < disorder_fraction < 0.95, (
        f"Label distribution is degenerate: {disorder_fraction:.2%} disordered."
    )

    model = DisorderNetV6()
    print(f"Training on {len(train_df)} sequences...")
    model.fit(train_df['sequence'].values, train_df['disorder_labels'].values)
    
    all_y_true = []
    all_y_pred = []
    
    print(f"Evaluating on {len(test_df)} sequences...")
    for _, row in test_df.iterrows():
        seq = row['sequence']
        labels = row['disorder_labels']
        preds = model.predict(seq)
        all_y_true.extend(labels)
        all_y_pred.extend(preds.flatten())
        
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Check 3: Verify test set has both classes
    assert len(np.unique(all_y_true)) == 2, "Test set contains only one class."
        
    auc = roc_auc_score(all_y_true, all_y_pred)
          
    print(f"Prediction stats: min={all_y_pred.min():.3f}, max={all_y_pred.max():.3f}, "
          f"mean={all_y_pred.mean():.3f}, std={all_y_pred.std():.3f}")
    print(f"AUC: {auc:.3f}")
    
    assert auc >= threshold, f'DisorderNet AUC {auc:.3f} below threshold {threshold} ({mode_str})'
        
    os.makedirs('benchmarks/disorder/results', exist_ok=True)
    with open('benchmarks/disorder/results/metrics.json', 'w') as f:
        json.dump({
            'auc_roc': auc,
            'mode': 'fast' if args.fast else 'full',
            'n_train': len(train_df),
            'n_test': len(test_df),
            'threshold': threshold
        }, f, indent=2)
        
if __name__ == '__main__':
    main()
