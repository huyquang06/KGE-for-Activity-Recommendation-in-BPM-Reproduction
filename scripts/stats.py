import os
from collections import Counter

DATA_DIR = r"D:\Học tập\HK1 Năm 26-27\Cơ sở các HTTT\Project_KGE\kge\data\bpmai_lastrev_caise_onlyAfter"

def read_triples(path):
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                print("Dòng lạ:", repr(line))
                continue
            triples.append(tuple(parts))
    return triples

splits = {s: read_triples(os.path.join(DATA_DIR, f"{s}.txt")) for s in ["train", "valid", "test"]}

entities, relations = set(), set()
for s, tr in splits.items():
    for h, r, t in tr:
        entities.update([h, t])
        relations.add(r)
    print(f"{s}: {len(tr)} triples")

print("Tổng triples:", sum(len(t) for t in splits.values()))
print("Số entity (train+valid+test):", len(entities))
print("Số relation:", len(relations), sorted(relations))

for s, tr in splits.items():
    print(f"Số triple theo relation ở {s}:", dict(Counter(r for _, r, _ in tr)))