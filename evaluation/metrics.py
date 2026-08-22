"""Evaluation metrics comparing a generated completion against groundtruth.

These only ever run *after* generation, purely for scoring -- never fed into
retrieval, selection, or generation.
"""
import re
from collections import Counter
from typing import List

import Levenshtein

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()


def edit_similarity(prediction: str, reference: str) -> float:
    """1 - (Levenshtein distance / max length), in [0, 1]. 1.0 for two empty strings."""
    pred, ref = prediction.strip(), reference.strip()
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0
    return 1.0 - (Levenshtein.distance(pred, ref) / max_len)


def extract_identifiers(text: str) -> List[str]:
    return _IDENTIFIER_RE.findall(text)


def identifier_f1(prediction: str, reference: str) -> float:
    """F1 over identifier tokens (multiset overlap, so repeated identifiers
    count multiple times on each side rather than collapsing to a set)."""
    pred_ids = Counter(extract_identifiers(prediction))
    ref_ids = Counter(extract_identifiers(reference))

    if not pred_ids and not ref_ids:
        return 1.0
    if not pred_ids or not ref_ids:
        return 0.0

    overlap = sum((pred_ids & ref_ids).values())
    precision = overlap / sum(pred_ids.values())
    recall = overlap / sum(ref_ids.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
