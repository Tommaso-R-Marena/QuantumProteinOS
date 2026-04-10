import os
import json
import requests
try:
    from qpos.chirality.auditor import ChiralFoldAuditor
    from qpos.data.pdb_parser import PDBParser
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

CHIRALITY_BENCHMARK_PDBS = [
    '1UBQ', '1L2Y', '1CRN', '1BRS', '1BPI', '1VII', '1GB1', '2HHB',
    '1MBN', '1TIM', '1HEW', '2LZM', '1AKE', '3LZT', '1TEN', '1IGD',
    '1HRC', '1PHT', '2CI2', '1CSP', '1FKB', '1HEL', '1IGT', '2PTN',
    '1RCB', '1SHF', '1STN', '1TOP', '2HVP', '1YCC', '4PTI'
]

def download_benchmark_pdbs(pdb_ids, output_dir):
    """Download multiple PDB files via direct RCSB HTTPS requests."""
    os.makedirs(output_dir, exist_ok=True)
    for pdb_id in pdb_ids:
        path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            try:
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                with open(path, 'w') as f:
                    f.write(r.text)
                print(f"Downloaded {pdb_id}")
            except Exception as e:
                print(f"Failed to download {pdb_id}: {e}")
        else:
            print(f"Using cached {pdb_id}")

def main():
    print("Running ChiralFold validation...")
    
    auditor = ChiralFoldAuditor()
    
    # Download testing fixtures
    fixture_dir = "tests/fixtures"
    download_benchmark_pdbs(CHIRALITY_BENCHMARK_PDBS, fixture_dir)
    
    correct_pcts = []
    perfect_structs = 0
    valid_structs = 0
    
    for pdb_id in CHIRALITY_BENCHMARK_PDBS:
        path = os.path.join(fixture_dir, f"{pdb_id}.pdb")
        if not os.path.exists(path):
            continue
            
        try:
            parser = PDBParser(path)
            report = auditor.audit(parser.structure)
            pct = report['chirality']['pct_correct']
            correct_pcts.append(pct)
            valid_structs += 1
            if pct == 100.0:
                perfect_structs += 1
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            
    if valid_structs > 0:
        mean_pct = sum(correct_pcts) / valid_structs
    else:
        mean_pct = 0.0
        
    print(f"Validation complete: Mean Pct={mean_pct:.1f}%, Perfect={perfect_structs}/{valid_structs}")
    
    os.makedirs('benchmarks/chirality/results', exist_ok=True)
    with open('benchmarks/chirality/results/audit_summary.json', 'w') as f:
        json.dump({
            'mean_chirality_pct_correct': mean_pct, 
            'n_structures_at_100pct': perfect_structs,
            'n_valid_structures': valid_structs
        }, f)

if __name__ == '__main__':
    main()
