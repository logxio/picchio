"""Optional Inspect AI task for scoring the exported recorded run dataset."""
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, scorer, accuracy
from inspect_ai.solver import solver


@solver
def recorded_run():
    async def solve(state, generate):
        return state
    return solve


@scorer(metrics=[accuracy()])
def independent_outcome():
    async def score(state, target):
        from humanscale.checks import recompute
        result=recompute(state.metadata['receipt'])
        scenario=state.metadata['scenario']
        ok=result['fake_green_caught'] if scenario==4 else result['genuine_fix']
        return Score(value=int(ok),answer=str(result),explanation='Recomputed from recorded receipt')
    return score


@task
def recorded_receipts():
    data=Path(__file__).resolve().parents[1]/'receipts/inspect.samples.jsonl'
    return Task(dataset=json_dataset(str(data)),solver=recorded_run(),scorer=independent_outcome())
