import os
import json
try:
    from qpos.chirality.auditor import ChiralFoldAuditor
    from qpos.data.pdb_parser import PDBParser
    from qpos.data.pdb_fetcher import PDBFetcher
except ImportError as e:
    raise NotImplementedError(f"Modular dependency unavailable: {e}")

CHIRALITY_BENCHMARK_PDBS = [
    '1UBQ', '1L2Y', '1CRN', '1BRS', '1BPI', '1VII', '1GB1', '2HHB',
    '1MBN', '1TIM', '1HEW', '2LZM', '1AKE', '3LZT', '1TEN', '1IGD',
    '1HRC', '1PHT', '2CI2', '1CSP', '1FKB', '1HEL', '1IGT', '2PTN',
    '1RCB', '1SHF', '1STN', '1TOP', '2HVP', '1YCC', '4PTI'
]

def main():
    print("Running ChiralFold validation...")
    
    auditor = ChiralFoldAuditor()
    
    # Download testing fixtures
    fixture_dir = "tests/fixtures"
    os.makedirs(fixture_dir, exist_ok=True)
    fetcher = PDBFetcher(output_dir=fixture_dir)
    
    for pdb_id in CHIRALITY_BENCHMARK_PDBS:
        fetcher.fetch_pdb(pdb_id)
        
    fixture_files = [os.path.join(fixture_dir, f"{pdb_id}.pdb") for pdb_id in CHIRALITY_BENCHMARK_PDBS]
    
    correct_pcts = []
    perfect_structs = 0
    
    for f in fixture_files:
        parser = PDBParser(f)
        report = auditor.audit(parser.structure)
        pct = report['chirality']['pct_correct']
        correct_pcts.append(pct)
        if pct == 100.0:
            perfect_structs += 1
            
    if len(correct_pcts) > 0:
        mean_pct = sum(correct_pcts) / len(correct_pcts)
    else:
        mean_pct = 0.0
        
    os.makedirs('benchmarks/chirality/results', exist_ok=True)
    with open('benchmarks/chirality/results/audit_summary.json', 'w') as f:
        json.dump({
            'mean_chirality_pct_correct': mean_pct, 
            'n_structures_at_100pct': perfect_structs
        }, f)

if __name__ == '__main__':
    main()
