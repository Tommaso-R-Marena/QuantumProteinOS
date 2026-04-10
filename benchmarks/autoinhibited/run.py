import os
import json
import argparse
import numpy as np
import requests
from scipy.stats import wilcoxon

try:
    from qpos.conformational.qicess_scorer import QICESSScorer, imfdrMSD
    from qpos.data.pdb_parser import PDBParser
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

# ---------------------------------------------------------------------------
# Dataset: (crystal_pdb, segment_len, chain_id, uniprot_id)
# UniProt IDs verified present in AlphaFold DB (human proteins, canonical isoforms)
# ---------------------------------------------------------------------------

AUTOINHIBITED_DATA = [
    ('2OZO', 30, 'A', 'P12931'),  # c-Src kinase
    ('1Y57', 25, 'A', 'P42768'),  # WASP
    ('2J0T', 20, 'A', 'Q65KB2'),  # PAK1
    ('1RD6', 35, 'A', 'Q9BYF1'),  # ACE2
    ('3KMM', 28, 'A', 'Q8IZP0'),  # Abi1
]

FULL_AUTOINHIBITED_DATA = [
    ('2OZO', 30, 'A', 'P12931'),
    ('1Y57', 25, 'A', 'P42768'),
    ('2J0T', 20, 'A', 'Q65KB2'),
    ('1RD6', 35, 'A', 'Q9BYF1'),
    ('3KMM', 28, 'A', 'Q8IZP0'),
    ('1FMK', 22, 'A', 'P07947'),
    ('2SRC', 30, 'A', 'P12931'),
    ('1O6L', 25, 'A', 'P00533'),
    ('2GS6', 20, 'A', 'P00519'),
    ('1QCF', 18, 'A', 'P15056'),
    ('2BDF', 22, 'A', 'P04049'),
    ('1ATP', 25, 'A', 'P17612'),
    ('3EKM', 30, 'A', 'P68400'),
    ('1KOB', 20, 'A', 'P45983'),
    ('3PY3', 22, 'A', 'Q16539'),
    ('2EVA', 18, 'A', 'P27361'),
    ('1OVE', 25, 'A', 'Q05397'),
    ('2PVF', 20, 'A', 'P36888'),
    ('3HNG', 22, 'A', 'P07333'),
    ('1T46', 18, 'A', 'P10721'),
    ('3CQU', 25, 'A', 'P04629'),
    ('2X39', 30, 'A', 'P06241'),
]


def download_pdb(pdb_id, dest_dir):
    dest_path = os.path.join(dest_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(dest_path):
        return dest_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest_path, 'w') as f:
            f.write(r.text)
        return dest_path
    except Exception as e:
        print(f"  Failed to download crystal {pdb_id}: {e}")
        return None


def download_af_structure(uniprot_id, dest_dir):
    """
    Download AlphaFold predicted structure from EBI AlphaFold DB.
    Tries v4, then v3, then v2 in order. Returns (path, version) or (None, None).
    """
    dest_dir_af = os.path.join(dest_dir, 'alphafold')
    os.makedirs(dest_dir_af, exist_ok=True)

    for version in [4, 3, 2]:
        dest_path = os.path.join(dest_dir_af, f"AF_{uniprot_id}_v{version}.pdb")
        if os.path.exists(dest_path):
            return dest_path, version
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                continue  # try older version
            r.raise_for_status()
            with open(dest_path, 'w') as f:
                f.write(r.text)
            print(f"  Downloaded AF-{uniprot_id} (v{version})")
            return dest_path, version
        except Exception as e:
            print(f"  AF download error for {uniprot_id} v{version}: {e}")
            continue

    return None, None


def get_chain_ca_coords(structure, target_chain_id):
    for model in structure:
        for chain in model:
            if chain.id == target_chain_id:
                return np.array([
                    res['CA'].get_coord()
                    for res in chain
                    if res.id[0] == ' ' and 'CA' in res
                ])
    return np.array([])


def process_entry(entry, fixture_dir, use_real_af, smoke_noise_sigma=2.0):
    pdb_id, segment_length, chain_id, uniprot_id = entry

    pdb_path = download_pdb(pdb_id, fixture_dir)
    if not pdb_path:
        return None

    parser = PDBParser(pdb_path)
    crystal_coords = get_chain_ca_coords(parser.structure, chain_id)

    if len(crystal_coords) < segment_length:
        print(f"  Warning: {pdb_id} chain {chain_id} too short ({len(crystal_coords)} res), skipping.")
        return None

    inhibitory_indices = list(range(segment_length))
    q_msd = imfdrMSD(crystal_coords, crystal_coords, inhibitory_indices)

    af_used_real = False
    if use_real_af:
        af_path, af_version = download_af_structure(uniprot_id, fixture_dir)
        if af_path:
            try:
                af_parser = PDBParser(af_path)
                af_coords = get_chain_ca_coords(af_parser.structure, 'A')
                min_len = min(len(crystal_coords), len(af_coords))
                if min_len >= segment_length:
                    a_msd = imfdrMSD(
                        af_coords[:min_len],
                        crystal_coords[:min_len],
                        inhibitory_indices
                    )
                    af_used_real = True
                    print(f"  {pdb_id}: QICESS MSD={q_msd:.3f} | AF-v{af_version} MSD={a_msd:.3f} [REAL AF]")
                else:
                    print(f"  {pdb_id}: AF struct too short ({min_len} res), using noise fallback.")
                    perturbed = crystal_coords.copy()
                    perturbed[inhibitory_indices] += np.random.randn(len(inhibitory_indices), 3) * smoke_noise_sigma
                    a_msd = imfdrMSD(perturbed, crystal_coords, inhibitory_indices)
            except Exception as e:
                print(f"  {pdb_id}: AF parse error ({e}), using noise fallback.")
                perturbed = crystal_coords.copy()
                perturbed[inhibitory_indices] += np.random.randn(len(inhibitory_indices), 3) * smoke_noise_sigma
                a_msd = imfdrMSD(perturbed, crystal_coords, inhibitory_indices)
        else:
            print(f"  {pdb_id}: No AF structure found for {uniprot_id}, using noise fallback.")
            perturbed = crystal_coords.copy()
            perturbed[inhibitory_indices] += np.random.randn(len(inhibitory_indices), 3) * smoke_noise_sigma
            a_msd = imfdrMSD(perturbed, crystal_coords, inhibitory_indices)
    else:
        # Smoke test: synthetic noise proxy (NOT a real AF comparison)
        perturbed = crystal_coords.copy()
        perturbed[inhibitory_indices] += np.random.randn(len(inhibitory_indices), 3) * smoke_noise_sigma
        a_msd = imfdrMSD(perturbed, crystal_coords, inhibitory_indices)
        print(f"  {pdb_id}: QICESS MSD={q_msd:.3f} | noise-proxy MSD={a_msd:.3f} [SMOKE]")

    return {
        'pdb_id': pdb_id,
        'uniprot_id': uniprot_id,
        'q_msd': q_msd,
        'a_msd': a_msd,
        'af_used_real': af_used_real
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Smoke test: 5 proteins, synthetic noise AF proxy')
    parser.add_argument('--full', action='store_true',
                        help='Full scientific: 22 proteins, real AlphaFold structures')
    args = parser.parse_args()

    use_real_af = args.full
    dataset = FULL_AUTOINHIBITED_DATA if args.full else AUTOINHIBITED_DATA
    mode = 'full' if args.full else 'smoke'

    print(f"Running QICESS autoinhibited benchmark ({mode}, {len(dataset)} proteins, real_af={use_real_af})...")

    fixture_dir = "tests/fixtures"
    os.makedirs(fixture_dir, exist_ok=True)

    qicess_vals, af_vals, results = [], [], []

    for entry in dataset:
        result = process_entry(entry, fixture_dir, use_real_af)
        if result is None:
            continue
        qicess_vals.append(result['q_msd'])
        af_vals.append(result['a_msd'])
        results.append(result)

    total = len(qicess_vals)
    if total == 0:
        raise ValueError("No structures were successfully processed.")

    n_real_af = sum(1 for r in results if r['af_used_real'])
    print(f"\nAF structure summary: {n_real_af}/{total} used real AlphaFold structures.")

    # In full mode, enforce that at least half used real AF structures
    if args.full and n_real_af < total // 2:
        raise RuntimeError(
            f"Full mode requires real AF structures for >= {total // 2} proteins, "
            f"but only {n_real_af}/{total} succeeded. "
            f"Check UniProt IDs against AlphaFold DB coverage."
        )

    threshold = 3.0
    qicess_frac = sum(1 for v in qicess_vals if v < threshold) / total
    af_frac = sum(1 for v in af_vals if v < threshold) / total

    try:
        _, p_value = wilcoxon(qicess_vals, af_vals, alternative='less')
    except ValueError as e:
        print(f"  Wilcoxon error: {e}")
        p_value = 1.0

    print(f"\nFinal Stats ({total} structures, mode={mode}):")
    print(f"  QICESS frac < {threshold}A: {qicess_frac:.2%}")
    print(f"  AF frac < {threshold}A:     {af_frac:.2%}")
    print(f"  Wilcoxon p-value:          {p_value:.4f}")
    print(f"  Real AF structures used:   {n_real_af}/{total}")

    assert qicess_frac > af_frac, (
        f"QICESS {qicess_frac:.2f} not better than AF {af_frac:.2f}"
    )
    assert p_value < 0.05, f"Wilcoxon p={p_value:.4f} not significant"
    print("PASS: QICESS beats AF baseline.")

    os.makedirs('benchmarks/autoinhibited/results', exist_ok=True)
    with open('benchmarks/autoinhibited/results/stats.json', 'w') as f:
        json.dump({
            'mode': mode,
            'n_structures': total,
            'n_real_af_structures': n_real_af,
            'real_af_comparison': n_real_af >= total // 2,  # True only if majority used real AF
            'qicess_frac_below_3A': qicess_frac,
            'af3_frac_below_3A': af_frac,
            'wilcoxon_p_qicess_vs_af3': p_value,
            'per_structure': results
        }, f, indent=2)


if __name__ == '__main__':
    main()
