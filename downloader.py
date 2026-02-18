import requests
import json
from pathlib import Path
from pprint import pprint
import pandas as pd

home = Path(__file__).parent
md_path = home / "metadata.jsonl"
d_path = home / "data"
base_url = "https://api.mavedb.org/api/v1"
term = "promoter"
limit = 100

d_path.mkdir(parents=True, exist_ok=True)
found_files = [str(f)[:-4] for f in d_path.iterdir()]

if md_path.exists():
    with open(md_path, "r", encoding="utf-8") as f:
        mdata = [json.loads(line) for line in f.readlines()]
else:
    mdata = []

if len(found_files) != len(mdata):
    raise ValueError("data mismatch")


search_params = {'search': term}
url = f"{base_url}/score-sets/search"
payload = {
    "text": term,
    "limit": limit
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    results = response.json()
    urns = {}
    for scoreset in results["scoreSets"]:
        urn = scoreset["urn"]
        
        mapped_url = f"https://api.mavedb.org/api/v1/score-sets/{urn}"
        response = requests.get(mapped_url)
        text = response.json()

        urns[urn] = {"type": [seq["targetSequence"]["sequenceType"] for seq in scoreset["targetGenes"]],
                     "metadata": text["extraMetadata"]}
else:
    raise ValueError("failed search")


to_query = []
for urn in urns:
    if len(urns[urn]["metadata"].keys()) > 0 and urn not in found_files:
        to_query.append(urn)
print(f"writing {to_query}")


for urn in to_query:
    scoreset_url = f"https://api.mavedb.org/api/v1/score-sets/{urn}/scores"
    csv = pd.read_csv(scoreset_url)
    csv.to_csv(d_path / f"{urn}.csv")
    with open(md_path, "a", encoding="utf-8") as f:
        text = json.dumps({urn: urns[urn]["metadata"]})
        f.write(text + "\n")