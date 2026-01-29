"""Microsoft Teams webhook notifier for HR summaries."""

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_MATCH_THRESHOLD = 0.7
DEFAULT_HIRE_THRESHOLD = 0.5


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_score(value: Any) -> Optional[float]:
    score = _to_float(value)
    if score is None:
        return None
    return score / 100.0 if score > 1 else score


def _candidate_name(candidate: Dict[str, Any]) -> str:
    resume = candidate.get("resume_data", {}) or {}
    return candidate.get("candidate_name") or resume.get("name") or candidate.get("resume_id", "Unknown")


def build_summary(
    job_id: str,
    ranked_candidates: List[Dict[str, Any]],
    fairness_metrics: Dict[str, Any],
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    hire_threshold: float = DEFAULT_HIRE_THRESHOLD,
) -> Dict[str, Any]:
    total = len(ranked_candidates)
    good_matches = 0
    bad_matches = 0
    hires = 0
    rejects = 0

    match_scores = []
    skill_scores = []

    for candidate in ranked_candidates:
        match_score = candidate.get("match_score")
        if match_score is None:
            match_score = candidate.get("match_data", {}).get("match_score")
        match_score = _normalize_score(match_score)

        if match_score is not None:
            match_scores.append(match_score)
            if match_score >= match_threshold:
                good_matches += 1
            else:
                bad_matches += 1

        skill_score = candidate.get("match_data", {}).get("skill_match")
        skill_score = _normalize_score(skill_score)
        if skill_score is not None:
            skill_scores.append(skill_score)

        decision = candidate.get("decision")
        if decision is None:
            hire_prob = _normalize_score(candidate.get("hire_probability", candidate.get("ranking_score")))
            decision = "HIRE" if hire_prob is not None and hire_prob >= hire_threshold else "REJECT"

        if decision == "HIRE":
            hires += 1
        else:
            rejects += 1

    avg_match = round(sum(match_scores) / len(match_scores), 4) if match_scores else None
    avg_skill = round(sum(skill_scores) / len(skill_scores), 4) if skill_scores else None

    top_candidates = []
    for candidate in ranked_candidates[:3]:
        match_score = candidate.get("match_score")
        if match_score is None:
            match_score = candidate.get("match_data", {}).get("match_score")
        match_score = _normalize_score(match_score)
        hire_prob = _normalize_score(candidate.get("hire_probability", candidate.get("ranking_score")))
        decision = candidate.get("decision") or ("HIRE" if hire_prob is not None and hire_prob >= hire_threshold else "REJECT")
        top_candidates.append({
            "name": _candidate_name(candidate),
            "rank": candidate.get("rank"),
            "match_score": match_score,
            "hire_probability": hire_prob,
            "decision": decision,
        })

    return {
        "job_id": job_id,
        "total_candidates": total,
        "good_matches": good_matches,
        "bad_matches": bad_matches,
        "hires": hires,
        "rejects": rejects,
        "avg_match_score": avg_match,
        "avg_skill_match": avg_skill,
        "fairness_metrics": fairness_metrics or {},
        "top_candidates": top_candidates,
    }


def build_teams_card(summary: Dict[str, Any]) -> Dict[str, Any]:
    fairness = summary.get("fairness_metrics", {})
    facts = [
        {"name": "Job ID", "value": summary.get("job_id", "")},
        {"name": "Total Candidates", "value": str(summary.get("total_candidates", 0))},
        {"name": "Good Matches", "value": str(summary.get("good_matches", 0))},
        {"name": "Bad Matches", "value": str(summary.get("bad_matches", 0))},
        {"name": "Hires", "value": str(summary.get("hires", 0))},
        {"name": "Rejects", "value": str(summary.get("rejects", 0))},
        {"name": "Avg Match Score", "value": str(summary.get("avg_match_score"))},
        {"name": "Avg Skill Match", "value": str(summary.get("avg_skill_match"))},
        {"name": "Impact Ratio", "value": str(fairness.get("impact_ratio"))},
        {"name": "DP Diff", "value": str(fairness.get("demographic_parity"))},
        {"name": "EO Diff", "value": str(fairness.get("equal_opportunity"))},
        {"name": "Fairness Mode", "value": str(fairness.get("fairness_mode", "OFF"))},
    ]

    top_lines = []
    for item in summary.get("top_candidates", []):
        top_lines.append(
            f"{item.get('rank')}. {item.get('name')} | match={item.get('match_score')} | hire_prob={item.get('hire_probability')} | {item.get('decision')}"
        )

    sections = [
        {"facts": facts, "markdown": True},
    ]

    if top_lines:
        sections.append({
            "text": "**Top Candidates**\n" + "\n".join(top_lines),
            "markdown": True,
        })

    return {
        "@type": "MessageCard",
        "@context": "http://schema.org/extensions",
        "summary": "Resume Screening Summary",
        "title": "Resume Screening Summary",
        "sections": sections,
    }


def post_to_teams(summary: Dict[str, Any], webhook_url: Optional[str] = None) -> None:
    url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
    if not url:
        raise RuntimeError("TEAMS_WEBHOOK_URL is not configured")

    payload = build_teams_card(summary)
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        status = resp.getcode()
        if status < 200 or status >= 300:
            raise RuntimeError(f"Teams webhook failed with status {status}")
