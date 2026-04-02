import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent / "data"

files = [f for f in data_path.iterdir()]

df = pd.read_csv(data_path / files[0])

print(df.columns)