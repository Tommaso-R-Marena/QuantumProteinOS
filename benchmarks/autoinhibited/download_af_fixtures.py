#!/usr/bin/env python
"""
Standalone script to pre-download all AlphaFold mmCIF fixtures locally.
Usage: python benchmarks/autoinhibited/download_af_fixtures.py

This populates tests/fixtures/alphafold/ so that run.py --full uses
cached files and never needs to contact the EBI server during CI.
"""
import os
import sys
import requests

# All UniProt IDs used in FULL_AUTOINHIBITED_DATA
UNIPROT_IDS = [
    'P12931', 'P42768', 'P16333', 'Q9BYF1', 'Q8IYH5',
    'P07947', 'P00533', 'P00519', 'P15056', 'P04049',
    'P17612', 'P19784', 'P45983', 'Q16539', 'P27361',
    'Q05397', 'P36888', 'P07333', 'P10721', 'P04629',
    'P06241',
]

OUT_DIR = os.path.join('tests', 'fixtures', 'alphafold')
BASE_URL = 'https://alphafold.ebi.ac.uk/files'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok, n_fail = 0, 0
    for uid in UNIPROT_IDS:
        out_path = os.path.join(OUT_DIR, f'AF_{uid}_v4.cif')
        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
            print(f'  {uid}: cached')
            n_ok += 1
            continue
        ok = False
        for version in [4, 3, 2]:
            url = f'{BASE_URL}/AF-{uid}-F1-model_v{version}.cif'
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200 and len(r.content) > 1000:
                    final_path = os.path.join(OUT_DIR, f'AF_{uid}_v{version}.cif')
                    with open(final_path, 'wb') as f:
                        f.write(r.content)
                    # Also write to _v4 canonical name so cache hits work
                    if version != 4:
                        with open(out_path, 'wb') as f:
                            f.write(r.content)
                    print(f'  {uid}: downloaded v{version} ({len(r.content)//1024} KB)')
                    ok = True
                    break
            except Exception as e:
                print(f'  {uid} v{version}: {e}')
        if ok:
            n_ok += 1
        else:
            print(f'  {uid}: FAILED all versions')
            n_fail += 1
    print(f'\nDone: {n_ok} OK, {n_fail} failed')
    if n_fail > len(UNIPROT_IDS) // 2:
        print('ERROR: more than half failed — check network access to alphafold.ebi.ac.uk', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
