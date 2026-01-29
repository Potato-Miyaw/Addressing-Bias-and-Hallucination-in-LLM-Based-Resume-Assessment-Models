"""Power Automate notifier for HR summaries with attachments."""

import base64
import json
import os
import urllib.request
from typing import Any, Dict, Optional


def _summary_text(summary: Dict[str, Any]) -> str:
    fairness = summary.get("fairness_metrics", {})
    lines = [
        f"Job {summary.get('job_id', '')}",
        f"Total: {summary.get('total_candidates', 0)}",
        f"Good Matches: {summary.get('good_matches', 0)} | Bad Matches: {summary.get('bad_matches', 0)}",
        f"Hires: {summary.get('hires', 0)} | Rejects: {summary.get('rejects', 0)}",
        f"Avg Match: {summary.get('avg_match_score')} | Avg Skill: {summary.get('avg_skill_match')}",
        f"Impact Ratio: {fairness.get('impact_ratio')} | DP Diff: {fairness.get('demographic_parity')} | EO Diff: {fairness.get('equal_opportunity')}",
    ]

    top_lines = []
    for item in summary.get("top_candidates", []):
        top_lines.append(
            f"{item.get('rank')}. {item.get('name')} | match={item.get('match_score')} | hire_prob={item.get('hire_probability')} | {item.get('decision')}"
        )

    if top_lines:
        lines.append("Top Candidates:")
        lines.extend(top_lines)

    return "\n".join(lines)


def send_to_power_automate(
    summary: Dict[str, Any],
    shortlist_csv: bytes,
    audit_pdf: bytes,
    url: Optional[str] = None,
) -> None:
    endpoint = url or os.getenv("POWER_AUTOMATE_URL")
    if not endpoint:
        raise RuntimeError("POWER_AUTOMATE_URL is not configured")

    fairness_mode = summary.get("fairness_metrics", {}).get("fairness_mode", "OFF")

    payload = {
        "job_id": summary.get("job_id"),
        "fairness_mode": fairness_mode,
        "summary_text": _summary_text(summary),
        "shortlist_csv_base64": base64.b64encode(shortlist_csv).decode("ascii"),
        "audit_pdf_base64": base64.b64encode(audit_pdf).decode("ascii"),
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            if status < 200 or status >= 300:
                raise RuntimeError(f"Power Automate call failed with status {status}")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"Power Automate call failed with status {exc.code}: {body}") from exc
