"""Cross-experiment comparison tables. Pure reporting -- takes already
computed summary dicts (from evaluation.baseline, evaluation.experiment's
run_experiment_v2 == LLM selection, evaluation.top2_selection, and
evaluation.random2_selection) and prints them side by side. Never runs or
modifies any experiment itself.
"""
from typing import Dict, Optional


def _fmt(value, spec: str = "{:.3f}") -> str:
    return spec.format(value) if value is not None else "n/a"


def print_four_way_comparison(
    no_selection_summary: Dict,
    llm_selection_summary: Dict,
    top2_summary: Dict,
    random2_summary: Dict,
) -> None:
    """Prints the no-selection baseline, LLM selection, top2_selection, and
    random2_selection summaries side by side, to answer whether LLM
    selection's improvement comes from intelligent candidate choice or
    just from shrinking the context size.
    """
    header = f"{'metric':<20} | {'no selection':>13} | {'LLM selection':>14} | {'top2':>8} | {'random2':>8}"
    print(header)
    print("-" * len(header))

    def row(label, no_sel, llm_sel, top2, random2, spec="{:.3f}"):
        print(
            f"{label:<20} | {_fmt(no_sel, spec):>13} | {_fmt(llm_sel, spec):>14} | "
            f"{_fmt(top2, spec):>8} | {_fmt(random2, spec):>8}"
        )

    row(
        "avg candidates",
        no_selection_summary.get("avg_candidate_count"),
        llm_selection_summary.get("avg_candidate_count"),
        top2_summary.get("avg_candidate_count"),
        random2_summary.get("avg_candidate_count"),
        "{:.2f}",
    )
    row(
        "avg selected",
        None,  # no-selection has no selection step
        llm_selection_summary.get("avg_selected_count"),
        top2_summary.get("avg_selected_count"),
        random2_summary.get("avg_selected_count"),
        "{:.2f}",
    )
    row(
        "exact match",
        no_selection_summary.get("exact_match_rate"),
        llm_selection_summary.get("exact_match_rate"),
        top2_summary.get("exact_match_rate"),
        random2_summary.get("exact_match_rate"),
    )
    row(
        "ES",
        no_selection_summary.get("avg_ES"),
        llm_selection_summary.get("avg_ES"),
        top2_summary.get("avg_ES"),
        random2_summary.get("avg_ES"),
    )
    row(
        "ID-F1",
        no_selection_summary.get("avg_ID_F1"),
        llm_selection_summary.get("avg_ID_F1"),
        top2_summary.get("avg_ID_F1"),
        random2_summary.get("avg_ID_F1"),
    )
    row(
        "avg time (s)",
        no_selection_summary.get("avg_generation_time"),
        llm_selection_summary.get("avg_generation_time"),
        top2_summary.get("avg_generation_time"),
        random2_summary.get("avg_generation_time"),
        "{:.2f}",
    )


def print_memory_ablation_comparison(
    no_selection_summary: Dict,
    llm_selection_summary: Dict,
    llm_selection_with_memory_summary: Dict,
    random_memory_summary: Optional[Dict] = None,
) -> None:
    """Prints no selection / LLM selection / LLM selection + structured
    repository memory (and, if given, LLM selection + random/noisy memory
    -- the ablation control) side by side, plus the memory-specific stats
    (memory-assisted selections, avg relationships retrieved, avg selected,
    selection count distribution) needed to see whether structured memory
    actually changed selection behavior, not just final metrics. Passing
    `random_memory_summary` answers whether it's specifically REAL
    structural evidence that helps, vs. any graph-shaped prompt text.
    """
    columns = [("no selection", no_selection_summary), ("LLM selection", llm_selection_summary), ("LLM + memory", llm_selection_with_memory_summary)]
    if random_memory_summary is not None:
        columns.append(("LLM + random memory", random_memory_summary))

    header = f"{'metric':<24} | " + " | ".join(f"{label:>13}" for label, _ in columns)
    print(header)
    print("-" * len(header))

    def row(label, key, spec="{:.3f}", skip_first=False):
        values = []
        for i, (_, summary) in enumerate(columns):
            value = None if (skip_first and i == 0) else summary.get(key)
            values.append(_fmt(value, spec))
        print(f"{label:<24} | " + " | ".join(f"{v:>13}" for v in values))

    row("avg candidates", "avg_candidate_count", "{:.2f}")
    row("avg selected", "avg_selected_count", "{:.2f}", skip_first=True)
    row("exact match", "exact_match_rate")
    row("ES", "avg_ES")
    row("ID-F1", "avg_ID_F1")
    row("avg generation time", "avg_generation_time", "{:.2f}")

    print()
    print("=== Memory-specific stats (LLM + memory) ===")
    print(f"  memory-assisted selections:      {llm_selection_with_memory_summary.get('memory_assisted_selections')}")
    print(
        "  avg memory relationships found:  "
        f"{_fmt(llm_selection_with_memory_summary.get('avg_memory_relationships_found'), '{:.2f}')}"
    )
    print(f"  avg selected candidates:         {_fmt(llm_selection_with_memory_summary.get('avg_selected_count'), '{:.2f}')}")
    dist = llm_selection_with_memory_summary.get("selection_count_distribution") or {}
    print(f"  selection count distribution:    0={dist.get('0')}  1={dist.get('1')}  2={dist.get('2')}  >=3={dist.get('>=3')}")

    if random_memory_summary is not None:
        print()
        print("=== Memory-specific stats (LLM + random memory) ===")
        print(f"  memory-assisted selections:      {random_memory_summary.get('memory_assisted_selections')}")
        print(
            "  avg memory relationships found:  "
            f"{_fmt(random_memory_summary.get('avg_memory_relationships_found'), '{:.2f}')}"
        )
        rdist = random_memory_summary.get("selection_count_distribution") or {}
        print(f"  selection count distribution:    0={rdist.get('0')}  1={rdist.get('1')}  2={rdist.get('2')}  >=3={rdist.get('>=3')}")
