"""Generate a realistic insurance dataset matching the Kaggle Medical Cost Personal Datasets schema."""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 1338

ages = np.random.randint(18, 65, n)
sexes = np.random.choice(['male', 'female'], n)
bmis = np.round(np.random.normal(30.6, 6.1, n), 2)
bmis = np.clip(bmis, 15.96, 53.13)
children = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
smokers = np.random.choice(['yes', 'no'], n, p=[0.205, 0.795])
regions = np.random.choice(['southwest', 'southeast', 'northwest', 'northeast'], n)

charges = []
for i in range(n):
    base = 1500 + ages[i] * 250 + children[i] * 400
    if bmis[i] >= 30:
        base += (bmis[i] - 30) * 150
    if smokers[i] == 'yes':
        base += 23000 + bmis[i] * 140
    if sexes[i] == 'male':
        base += 200
    noise = np.random.normal(0, base * 0.1)
    charges.append(round(max(1121.87, base + noise), 2))

df = pd.DataFrame({
    'age': ages, 'sex': sexes, 'bmi': bmis,
    'children': children, 'smoker': smokers,
    'region': regions, 'charges': charges
})
df.to_csv("C:\\Users\\USER\\Desktop\\insurance.csv", index=False)
print(f"Dataset saved: {len(df)} rows")
print(df.head())
