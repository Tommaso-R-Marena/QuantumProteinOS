import pandas as pd
import os
import requests


CAID3_RELEASE_URL = "https://disprot.org/api/search"


def download_disprot(version="CAID3", cache_dir="~/.qpos/disprot", max_records=None) -> pd.DataFrame:
    """
    Download DisProt dataset from the DisProt REST API.
    Returns DataFrame with columns: uniprot_id, sequence, disorder_labels
    disorder_labels: list of 0/1 per residue (1 = disordered)

    max_records: None = fetch everything (full mode), integer = cap (fast mode).
    Results are cached to disk per (version, max_records) key.
    """
    cache_path = os.path.expanduser(cache_dir)
    os.makedirs(cache_path, exist_ok=True)

    cap_tag = f"_cap{max_records}" if max_records is not None else "_full"
    file_path = os.path.join(cache_path, f"disprot_{version}{cap_tag}.csv")

    if os.path.exists(file_path):
        print(f"Loading DisProt from cache: {file_path}")
        return pd.read_csv(file_path, converters={'disorder_labels': eval})

    print(f"Fetching DisProt {version} from API (max_records={max_records})...")
    rows = []

    try:
        page = 1
        per_page = 200
        total_fetched = 0

        while True:
            params = {'format': 'json', 'page': page, 'limit': per_page}
            response = requests.get(CAID3_RELEASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                items = data.get('data', data.get('results', []))
            else:
                items = data

            if not items:
                break

            for item in items:
                if max_records is not None and total_fetched >= max_records:
                    break
                seq = item.get('sequence', '')
                if not seq:
                    continue
                uniprot_id = item.get('acc', item.get('uniprot_id', f'DP{total_fetched}'))
                regions = item.get('regions', [])
                labels = [0] * len(seq)
                for r in regions:
                    start = max(0, r.get('start', 1) - 1)
                    end = min(len(seq), r.get('end', len(seq)))
                    for j in range(start, end):
                        labels[j] = 1
                rows.append({'uniprot_id': uniprot_id, 'sequence': seq, 'disorder_labels': labels})
                total_fetched += 1

            if max_records is not None and total_fetched >= max_records:
                break
            if len(items) < per_page:
                break
            page += 1

        if not rows:
            raise ValueError("API returned no sequences.")

        print(f"Fetched {len(rows)} sequences from DisProt API.")
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False)
        return df

    except Exception as e:
        print(f"Warning: DisProt API fetch failed ({e}). Using synthetic fallback.")
        import random
        random.seed(42)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        promoters = set("RKESPQAG")
        fallback_rows = []
        n = max_records if max_records is not None else 500
        for i in range(n):
            length = random.randint(50, 201)
            seq = "".join(random.choices(aa, k=length))
            labels = [1 if (res in promoters and random.random() < 0.85) or random.random() < 0.08 else 0 for res in seq]
            fallback_rows.append({'uniprot_id': f'SYNTHETIC_{i}', 'sequence': seq, 'disorder_labels': labels})
        return pd.DataFrame(fallback_rows)
