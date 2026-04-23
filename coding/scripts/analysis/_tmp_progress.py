from pathlib import Path
import json
from collections import Counter

base = list(Path('H:/').iterdir())[0]
mac = [x for x in base.iterdir() if 'Mac' in x.name][0]
cj = mac / 'data flow' / 'coding_judge'

for name in ['judge_results_pro.jsonl', 'judge_results_flash.jsonl']:
    out = cj / name
    if not out.exists():
        print(f"{name}: not started yet")
        continue
    lines = []
    with out.open('r', encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if l:
                try:
                    lines.append(json.loads(l))
                except json.JSONDecodeError:
                    pass  # skip corrupt lines from concurrent writes
    st = Counter(r['judge']['status'] for r in lines)
    print(f"\n{name}: {len(lines)} items done")
    for s, c in st.most_common():
        pct = c / len(lines) * 100
        print(f"  {s:30s} {c:>5d}  ({pct:5.1f}%)")
