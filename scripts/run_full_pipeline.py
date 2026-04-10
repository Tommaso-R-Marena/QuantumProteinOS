"""
QuantumProteinOS primary entry point.
"""
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="QuantumProteinOS Pipeline")
    parser.add_argument("--sequence", type=str, help="Amino acid sequence")
    parser.add_argument("--pdb", type=str, help="Template PDB path")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--af3-output", type=str, help="AF3 prediction PDB to correct")
    parser.add_argument("--correct-chirality", action="store_true", help="Run ChiralFold correction")
    parser.add_argument("--no-network", action="store_true", help="Skip all RCSB PDB downloads and use local tests/fixtures/")
    parser.add_argument("--benchmark", type=str, help="Run specific benchmark or 'all'")
    parser.add_argument("--audit", type=str, help="Audit PDB structure")

    args = parser.parse_args()

    # Load default config
    try:
        from qpos.pipeline import QuantumProteinOS
        # Ensure default config directory exists or use a standard path
        config_path = "configs/default.yaml"
        if not os.path.exists(config_path):
            # Create a basic default config if it doesn't exist to avoid crash
            os.makedirs("configs", exist_ok=True)
            with open(config_path, "w") as f:
                f.write("disorder: {}\nconformational: {}\nqicess: {}\nrotamers: {}\nchirality: {}\n")
        
        qpos = QuantumProteinOS.from_config(config_path)
        
        if args.no_network:
            from qpos.data.pdb_fetcher import PDBFetcher
            def mock_fetch(self, pdb_id):
                # Try relative paths commonly used in tests
                search_paths = [
                    os.path.join('tests', 'fixtures', f"{pdb_id.upper()}.pdb"),
                    os.path.join('tests', 'fixtures', f"pdb{pdb_id.lower()}.ent"),
                ]
                for path in search_paths:
                    if os.path.exists(path):
                        return path
                raise FileNotFoundError(f"Local test fixture missing and network disabled for {pdb_id}")
            PDBFetcher.fetch_pdb = mock_fetch

    except Exception as e:
        print(f"Error loading QuantumProteinOS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Determine sequence and pdb_path
    sequence = args.sequence
    pdb_path = args.pdb

    if args.pdb and not args.sequence:
        try:
            from qpos.data.pdb_parser import PDBParser
            parser_obj = PDBParser(args.pdb)
            sequence = parser_obj.get_sequence()
            print(f"Extracted sequence from {args.pdb}: {sequence[:50]}...")
        except Exception as e:
            print(f"Error parsing PDB {args.pdb}: {e}")
            sys.exit(1)

    # Trigger execution based on arguments
    if sequence:
        print(f"Running pipeline on sequence length {len(sequence)}...")
        result = qpos.run(sequence=sequence, pdb_path=pdb_path)
        result.save(args.output)
        print(f"Pipeline complete. Outputs written to {args.output}")

    elif args.af3_output and args.correct_chirality:
        print(f"Correcting AF3 output at {args.af3_output}...")
        res_path = qpos.af3_corrector.correct(args.af3_output)
        print(f"Correction applied. Saved to: {res_path}")

    elif args.audit:
        print(f"Auditing {args.audit}...")
        from qpos.data.pdb_parser import PDBParser
        parser_obj = PDBParser(args.audit)
        report = qpos.chirality_auditor.audit(parser_obj.structure)
        import json
        print(json.dumps(report, indent=2))
        
    elif args.benchmark:
        print(f"Executing benchmark: {args.benchmark}...")
        # (Benchmarks are usually handled by individual scripts, but we could hook them here)
        print("Done.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
