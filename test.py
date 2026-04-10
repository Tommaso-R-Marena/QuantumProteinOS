import numpy as np
from qpos.disorder.ensemble import DisorderNetV6

rows = []
for i in range(500):
    seq = "A" * 100
    labels = [1 if j < 20 else 0 for j in range(100)]
    rows.append({'uniprot_id': f'DP{i}', 'sequence': seq, 'disorder_labels': labels})

import pandas as pd
df = pd.DataFrame(rows)
train_df = df.iloc[:400]
test_df = df.iloc[400:]

model = DisorderNetV6()
model.fit(train_df['sequence'].values, train_df['disorder_labels'].values)

seq = test_df.iloc[0]['sequence']
preds = model.predict(seq)
print('preds:', preds[:5], '...', preds[-5:])
print('min', preds.min(), 'max', preds.max(), 'std', preds.std())
