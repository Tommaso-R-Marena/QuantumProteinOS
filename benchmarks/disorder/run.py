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
    parser.add_argument('--fast', action='store_true', help='500-seq subset smoke test (CI per-push)')
    parser.add_argument('--full', action='store_true', help='Full DisProt CAID3 evaluation (monthly)')
    args = parser.parse_args()

    if args.full:
        # Full mode: fetch ALL DisProt sequences, no cap
        # This is the number that goes in the paper.
        print("Running DisorderNet benchmark (full — entire DisProt CAID3)...")
        df = download_disprot(version="CAID3", max_records=None)
        threshold = 0.82
        mode_str = "full mode"
    else:
        # Fast / smoke test: cap at 500 sequences
        # Purpose: confirm pipeline runs and AUC >= 0.65. NOT the paper number.
        print("Running DisorderNet benchmark (fast — 500-seq smoke test)...")
        df = download_disprot(version="CAID3", max_records=500)
        threshold = 0.65
        mode_str = "fast mode (smoke test)"

    print(f"Dataset size: {len(df)} sequences")

    def parse_labels(lbl):
        if isinstance(lbl, str):
            import ast
            try:
                return ast.literal_eval(lbl)
            except Exception:
                return [int(x) for x in lbl.strip('[]').split(',')]
        return lbl

    df['disorder_labels'] = df['disorder_labels'].apply(parse_labels)
    df['pct_disordered'] = df['disorder_labels'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df['is_mostly_disordered'] = (df['pct_disordered'] > 0.3).astype(int)

    if args.fast:
        if len(df) > 500:
            df, _ = train_test_split(df, train_size=500, random_state=42, stratify=df['is_mostly_disordered'])
        train_df, test_df = train_test_split(
            df, test_size=150, random_state=42,
            stratify=df['is_mostly_disordered']
        )
    else:
        # Full mode: hard fail if dataset is too small to be scientifically valid
        if len(df) < 1000:
            raise RuntimeError(
                f"Full-mode DisProt download returned only {len(df)} sequences. "
                f"Expected >= 1000. Check network access or API response."
            )
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df['is_mostly_disordered']
        )

    # Check 1: Data leakage
    train_ids = set(train_df['uniprot_id'].tolist())
    test_ids = set(test_df['uniprot_id'].tolist())
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} sequences in both train and test"

    # Check 2: Label distribution
    all_train_labels = np.concatenate(train_df['disorder_labels'].tolist())
    disorder_fraction = all_train_labels.mean()
    print(f"Training label distribution: {disorder_fraction:.2%} disordered")
    assert 0.05 < disorder_fraction < 0.95, f"Degenerate label distribution: {disorder_fraction:.2%}"

    model = DisorderNetV6()
    print(f"Training on {len(train_df)} sequences...")
    model.fit(train_df['sequence'].values, train_df['disorder_labels'].values)

    all_y_true, all_y_pred = [], []
    print(f"Evaluating on {len(test_df)} sequences...")
    for _, row in test_df.iterrows():
        preds = model.predict(row['sequence'])
        all_y_true.extend(row['disorder_labels'])
        all_y_pred.extend(preds.flatten())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Check 3: Both classes in test set
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
            'mode': 'full' if args.full else 'fast',
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_total_sequences': len(df),
            'threshold': threshold,
            'dataset': 'DisProt CAID3'
        }, f, indent=2)


if __name__ == '__main__':
    main()
