"""Summarize common evaluation outputs; no algorithm-specific reward mixing."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);a=p.parse_args()
rows=[]
for run in a.runs:
    reports=[json.loads(f.read_text()) for f in sorted(run.glob('eval-*/evaluation.json'))]
    episodes=[e for r in reports for e in r['episodes']]
    n=len(episodes)
    elapsed=run/"training_elapsed_seconds.txt"
    rows.append(dict(run=str(run),evaluated_episodes=n,
        training_elapsed_seconds=float(elapsed.read_text()) if elapsed.exists() else None,
        seeds=[r['seed'] for r in reports],
        mean_return=sum(e['return_value'] for e in episodes)/n if n else None,
        mean_goals=sum(e['goals'] for e in episodes)/n if n else None,
        any_goal_success_rate=sum(e['goals']>0 for e in episodes)/n if n else None,
        all_goals_success_rate=sum(e['all_goals'] for e in episodes)/n if n else None))
print(json.dumps(rows,indent=2))
