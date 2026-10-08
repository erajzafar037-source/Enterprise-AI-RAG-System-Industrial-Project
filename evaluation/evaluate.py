import json,sys
from pathlib import Path
def score(answer,expected_terms):
    a=answer.lower()
    return sum(t.lower() in a for t in expected_terms)/len(expected_terms)
def main(path="evaluation/predictions.example.jsonl"):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    value=sum(score(r["answer"],r["expected_terms"]) for r in rows)/len(rows) if rows else 0
    print(f"term_coverage={value:.3f}")
    return 0 if value>=0.70 else 1
if __name__=="__main__": sys.exit(main(*sys.argv[1:]))
