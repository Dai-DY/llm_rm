"""
Rule-based post-processing utilities for explicit prompt constraints.

This module detects hard formatting requirements in the prompt, such as JSON
format, maximum word count, maximum sentence count, exact bullet count, or no
repeated words. It scores response A and response B against the detected
constraints and optionally adjusts the final ensemble probabilities with a
small conservative logit update.

The rule checker is not a semantic judge. It is only intended to correct clear
instruction-following violations after the neural ensemble has produced its
probabilities.
"""

import json
import re
from collections import Counter

import torch
import torch.nn.functional as F


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _text(value):
    return "" if value is None else str(value)


def count_words(text):
    return len(WORD_RE.findall(_text(text)))


def count_sentences(text):
    parts = [part for part in re.split(r"[.!?。！？]+", _text(text)) if part.strip()]
    return len(parts)


def has_unique_words(text):
    words = [word.lower() for word in WORD_RE.findall(_text(text))]
    return len(words) == len(set(words))


def is_valid_json(text):
    try:
        json.loads(_text(text))
    except Exception:
        return False
    return True


def _extract_code_blocks(text):
    return re.findall(r"```(?:json|JSON)?\s*(.*?)```", _text(text), flags=re.DOTALL)


def _balanced_json_candidates(text):
    text = _text(text)
    pairs = [("[", "]"), ("{", "}")]
    candidates = []
    for open_char, close_char in pairs:
        starts = [idx for idx, char in enumerate(text) if char == open_char]
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : idx + 1])
                        break
    return candidates


def contains_valid_json(text):
    if is_valid_json(text):
        return True
    for block in _extract_code_blocks(text):
        if is_valid_json(block.strip()):
            return True
    for candidate in _balanced_json_candidates(text):
        if is_valid_json(candidate.strip()):
            return True
    return False


def requires_strict_json(prompt):
    lowered = _text(prompt).lower()
    return bool(
        re.search(
            r"\b(?:only|just)\s+(?:return|respond|output|provide)?\s*(?:valid\s+)?json\b|"
            r"\b(?:return|respond|output|provide)\s+(?:only|just)\s+(?:valid\s+)?json\b|"
            r"\bno explanations?\b|\bwithout explanations?\b|\bdo not (?:include|add|write) explanations?\b",
            lowered,
        )
    )


def count_bullets(text):
    count = 0
    for line in _text(text).splitlines():
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            count += 1
    return count


def extract_constraints(prompt):
    prompt_text = _text(prompt)
    lowered = prompt_text.lower()
    constraints = []

    if re.search(
        r"(?:each|every)\s+word\b.*\b(?:once|only once|no repeats?|without repeating|not repeat)\b|"
        r"\bno repeated words?\b|\bdo not repeat (?:any )?words?\b",
        lowered,
    ):
        constraints.append({"type": "unique_words"})

    word_match = re.search(
        r"\b(?:under|less than|no more than|within|at most|max(?:imum of)?)\s+(\d+)\s+words?\b",
        lowered,
    )
    if word_match:
        constraints.append({"type": "max_words", "value": int(word_match.group(1))})

    sentence_match = re.search(
        r"\b(?:under|less than|no more than|within|at most|max(?:imum of)?)\s+(\d+)\s+sentences?\b",
        lowered,
    )
    if sentence_match:
        constraints.append({"type": "max_sentences", "value": int(sentence_match.group(1))})

    bullet_match = re.search(
        r"\b(?:exactly\s+)?(\d+)\s+(?:bullet points?|bullets|list items?)\b",
        lowered,
    )
    if bullet_match:
        constraints.append({"type": "bullet_count", "value": int(bullet_match.group(1))})

    if re.search(r"\b(?:valid\s+)?json\b|\bjson\s+(?:format|object|array)\b", lowered):
        constraints.append({"type": "json", "strict": requires_strict_json(prompt_text)})

    return constraints


def score_response_against_constraints(response, constraints):
    score = 0.0
    details = []
    for constraint in constraints:
        kind = constraint["type"]
        if kind == "unique_words":
            passed = has_unique_words(response)
        elif kind == "max_words":
            passed = count_words(response) <= constraint["value"]
        elif kind == "max_sentences":
            passed = count_sentences(response) <= constraint["value"]
        elif kind == "bullet_count":
            passed = count_bullets(response) == constraint["value"]
        elif kind == "json":
            passed = is_valid_json(response) if constraint.get("strict") else contains_valid_json(response)
        else:
            continue

        score += 1.0 if passed else -1.0
        details.append({"type": kind, "passed": passed, "value": constraint.get("value")})

    return score, details


def score_pair(prompt, response_a, response_b):
    constraints = extract_constraints(prompt)
    if not constraints:
        return 0.0, 0.0, []

    score_a, details_a = score_response_against_constraints(response_a, constraints)
    score_b, details_b = score_response_against_constraints(response_b, constraints)
    return score_a, score_b, [{"constraints": constraints, "a": details_a, "b": details_b}]


def collect_rule_adjustments(frame):
    rows = []
    matched_count = 0
    for row in frame.itertuples(index=False):
        score_a, score_b, details = score_pair(row.prompt, row.response_a, row.response_b)
        if details:
            matched_count += 1
        rows.append((score_a, score_b, details))
    return rows, matched_count


def apply_rule_adjustment(probs, frame, rule_weight=0.8, tie_penalty=0.2, return_report=False):
    adjustments, matched_count = collect_rule_adjustments(frame)
    if matched_count == 0 or rule_weight <= 0:
        if return_report:
            return probs, {"matched_count": matched_count, "total": len(frame), "changed_count": 0}
        return probs

    logits = torch.log(probs.clamp_min(1e-8)).clone()
    before = probs.argmax(dim=-1)

    for idx, (score_a, score_b, _) in enumerate(adjustments):
        delta = score_a - score_b
        if delta == 0:
            continue
        logits[idx, 0] += rule_weight * delta
        logits[idx, 1] -= rule_weight * delta
        logits[idx, 2] -= tie_penalty * rule_weight * abs(delta)

    adjusted = F.softmax(logits, dim=-1)
    if not return_report:
        return adjusted

    after = adjusted.argmax(dim=-1)
    return adjusted, {
        "matched_count": matched_count,
        "total": len(frame),
        "changed_count": int((before != after).sum().item()),
    }
