"""Measure routing and answer quality against the gold set.

Run:  uv run python scripts/run_eval.py
      uv run python scripts/run_eval.py --only p02,o01
      uv run python scripts/run_eval.py --fresh

Resumable on purpose. The free tier allows 20 model calls per day and one full pass costs
roughly 60, so a run will hit the quota part way through. Results are appended to
`data/eval/results.jsonl` as they are produced; re-running skips ids already recorded, so the
set can be finished across several days or several models without repeating work.

Write cases mutate the database. The whole run works on a temporary copy, so `data/shop.db`
is never touched.

Each row records the model that produced it. Finishing a run across two models is fine for
completing the set, but the scores are then not a clean comparison — check the `model` column
before drawing conclusions from a single number.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage  # noqa: E402

from src.agents.graph import build_graph, open_checkpointer  # noqa: E402
from src.config import settings  # noqa: E402
from src.tools import db  # noqa: E402

GOLD = Path("data/eval/gold.jsonl")
RESULTS = Path("data/eval/results.jsonl")
CUSTOMER_ID = "C026"  # the customer who owns order 1001 in the seeded data


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


SPECIALIST_NODES = {"policy_agent", "order_agent", "action_agent"}


def run_case(graph, case: dict) -> dict:
    """Run one question, recording which specialist actually handled it.

    The route is *observed* by streaming node execution, not inferred from the final state.
    Inferring it from citations was wrong: a correct policy answer that happens not to quote
    a filename produces no citations and was scored as a routing failure.
    """
    started = time.time()
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    visited: list[str] = []

    for step in graph.stream(
        {"messages": [HumanMessage(case["question"])], "customer_id": CUSTOMER_ID, "step_count": 0},
        config=config,
    ):
        visited.extend(node for node in step if node in SPECIALIST_NODES)

    snapshot = graph.get_state(config)
    state = snapshot.values

    if snapshot.next or state.get("pending_action"):
        answer = (state.get("pending_action") or {}).get("description", "")
        route = "action_agent"
        sources: list[str] = []
    else:
        answer = "\n".join(
            m.text for m in state.get("messages", []) if getattr(m, "text", "").strip()
        )
        route = visited[-1] if visited else "finish"
        sources = sorted({c["source"] for c in state.get("citations", [])})

    lowered = answer.lower()
    return {
        "id": case["id"],
        "model": settings.llm_model,
        "question": case["question"],
        "expected_route": case["route"],
        "actual_route": route,
        "visited": visited,
        "route_ok": route == case["route"],
        "expected_sources": case["sources"],
        "actual_sources": sources,
        "source_ok": (not case["sources"]) or bool(set(case["sources"]) & set(sources)),
        "content_ok": all(term.lower() in lowered for term in case["must_contain"]),
        "answer": answer[:400],
        "seconds": round(time.time() - started, 1),
    }


def report(rows: list[dict]) -> None:
    if not rows:
        print("No results yet.")
        return

    print(f"\n{'id':6} {'route':14} {'ok':4} {'src':4} {'txt':4}  question")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: r["id"]):
        mark = lambda ok: " ok " if ok else "FAIL"  # noqa: E731
        print(
            f"{r['id']:6} {r['actual_route']:14} {mark(r['route_ok'])} "
            f"{mark(r['source_ok'])} {mark(r['content_ok'])}  {r['question'][:38]}"
        )

    n = len(rows)
    routing = sum(r["route_ok"] for r in rows)
    sourcing = sum(r["source_ok"] for r in rows)
    content = sum(r["content_ok"] for r in rows)
    print("-" * 78)
    print(f"cases scored      {n}/20")
    print(f"routing accuracy  {routing}/{n} ({routing / n:.0%})   target >= 90%")
    print(f"correct source    {sourcing}/{n} ({sourcing / n:.0%})")
    print(f"answer contains   {content}/{n} ({content / n:.0%})")

    failed = [r for r in rows if not (r["route_ok"] and r["source_ok"] and r["content_ok"])]
    if failed:
        print(f"\n{len(failed)} case(s) to look at:")
        for r in failed:
            print(f"  [{r['id']}] expected {r['expected_route']}, got {r['actual_route']}")
            print(f"        {r['answer'][:110]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated ids, e.g. p02,o01")
    parser.add_argument("--fresh", action="store_true", help="discard previous results")
    parser.add_argument("--report", action="store_true", help="print the report and exit")
    args = parser.parse_args()

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and RESULTS.exists():
        RESULTS.unlink()

    if args.report:
        report(load(RESULTS))
        return

    cases = load(GOLD)
    if args.only:
        wanted = {i.strip() for i in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]

    done = {r["id"] for r in load(RESULTS)}
    todo = [c for c in cases if c["id"] not in done]

    print(f"gold set {len(load(GOLD))} cases | already scored {len(done)} | running {len(todo)}")
    if not todo:
        report(load(RESULTS))
        return

    # Write cases change data. Work on a copy so data/shop.db stays reproducible.
    tmp = Path(tempfile.mkdtemp()) / "shop.db"
    shutil.copy(settings.sqlite_path, tmp)
    db.settings.sqlite_path = tmp

    # A checkpointer is required: the run observes each case with `get_state`, and the write
    # cases pause on `interrupt()`. It is thrown away with the temporary database.
    graph = build_graph(checkpointer=open_checkpointer(tmp.parent / "eval.sqlite"))
    for case in todo:
        try:
            row = run_case(graph, case)
        except Exception as exc:  # noqa: BLE001 — quota and network failures stop the run
            print(f"\n[{case['id']}] stopped: {type(exc).__name__}: {str(exc)[:120]}")
            print("Progress is saved. Re-run to continue from here.")
            break
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        flag = "ok  " if row["route_ok"] else "FAIL"
        print(f"  [{row['id']}] {flag} {row['actual_route']:14} {row['seconds']}s")

    report(load(RESULTS))


if __name__ == "__main__":
    main()
