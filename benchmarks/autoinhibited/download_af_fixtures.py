#!/usr/bin/env python
"""
Standalone script to pre-download all AlphaFold mmCIF fixtures locally.
Usage: python benchmarks/autoinhibited/download_af_fixtures.py

This populates tests/fixtures/alphafold/ so that run.py --full uses
cached files and never needs to contact the EBI server during CI.

NOTE (April 2026): Updated to AFDB v6 REST API.
Old direct-file URLs (AF-{id}-F1-model_v4.cif) return 404 since AFDB v6
launched Oct 2025. Sunset date for old API: 25 June 2026 (EBI announcement).
New method: GET /api/prediction/{uniprot_id} -> entry[0]['cifUrl'] -> download.
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
AF_API_BASE = 'https://alphafold.ebi.ac.uk/api/prediction'


def download_af_v6(uniprot_id, out_dir):
    """
    Download AF mmCIF via AFDB v6 REST API.
    Returns (path, version_str) or (None, None) on failure.
    """
    out_path = os.path.join(out_dir, f'AF_{uniprot_id}_v6.cif')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path, 'v6-cached'

    # Step 1: API lookup
    try:
        r = requests.get(f'{AF_API_BASE}/{uniprot_id}', timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f'  {uniprot_id}: API lookup failed — {e}')
        return None, None

    if not data:
        print(f'  {uniprot_id}: API returned empty response')
        return None, None

    cif_url = data[0].get('cifUrl') or data[0].get('mmcifUrl')
    if not cif_url:
        print(f'  {uniprot_id}: no cifUrl in API response. Keys: {list(data[0].keys())}')
        return None, None

    # Extract version for logging
    version_str = 'v6'
    for part in cif_url.split('-'):
        if part.startswith('v') and part.rstrip('.cif').isdigit():
            version_str = part.rstrip('.cif')
            break

    # Step 2: Download
    try:
        r2 = requests.get(cif_url, timeout=60)
        r2.raise_for_status()
        with open(out_path, 'w') as f:
            f.write(r2.text)
        return out_path, version_str
    except Exception as e:
        print(f'  {uniprot_id}: CIF download failed — {e}')
        return None, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok, n_fail = 0, 0

    print(f'Downloading {len(UNIPROT_IDS)} AF structures via AFDB v6 REST API...')
    print(f'Output: {OUT_DIR}\n')

    for uid in UNIPROT_IDS:
        path, version = download_af_v6(uid, OUT_DIR)
        if path:
            size_kb = os.path.getsize(path) // 1024
            print(f'  {uid}: {version} ({size_kb} KB) -> {os.path.basename(path)}')
            n_ok += 1
        else:
            print(f'  {uid}: FAILED')
            n_fail += 1

    print(f'\nDone: {n_ok}/{len(UNIPROT_IDS)} OK, {n_fail} failed')
    if n_fail > len(UNIPROT_IDS) // 2:
        print('ERROR: more than half failed — check network access to alphafold.ebi.ac.uk',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
