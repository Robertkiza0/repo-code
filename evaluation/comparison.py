"""Cross-experiment comparison tables. Pure reporting -- takes already
computed summary dicts (from evaluation.baseline, evaluation.experiment's
run_experiment_v2 == LLM selection, evaluation.top2_selection, and
evaluation.random2_selection) and prints them side by side. Never runs or
modifies any experiment itself.
"""
from typing import Dict


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
