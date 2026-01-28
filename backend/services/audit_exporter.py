"""Audit export helpers for shortlist CSV and audit PDF."""

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _verification_summary(ranked_candidates: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    summary = {
        "total_candidates": len(ranked_candidates),
        "flagged_candidates": 0,
        "total_flagged_claims": 0,
        "verdict_counts": {},
    }
    flagged_lines: List[str] = []

    for candidate in ranked_candidates:
        verification = candidate.get("verification") or {}
        verification_status = candidate.get("verification_status")

        if isinstance(verification, dict):
            verdict = verification.get("verdict") or verification_status or "UNKNOWN"
            flagged_list = verification.get("flagged") or []
            flagged_count = verification.get("flagged_claims")
            if isinstance(flagged_count, int):
                flagged_total = flagged_count
            else:
                flagged_total = len(flagged_list) if isinstance(flagged_list, list) else 0

            summary["total_flagged_claims"] += flagged_total
            if flagged_total > 0 or verdict == "CONTAINS_HALLUCINATIONS":
                summary["flagged_candidates"] += 1

            for item in flagged_list[:5]:
                field = item.get("field", "unknown") if isinstance(item, dict) else "unknown"
                resume_id = candidate.get("resume_id", "unknown")
                flagged_lines.append(f"{resume_id}: {field}")
        else:
            verdict = verification_status or "UNKNOWN"

        summary["verdict_counts"][verdict] = summary["verdict_counts"].get(verdict, 0) + 1

    return summary, flagged_lines


def build_shortlist_csv(
    ranked_candidates: List[Dict[str, Any]],
    fairness_mode: str,
    hire_threshold: float,
) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "resume_id",
            "candidate_name",
            "rank",
            "hire_probability",
            "decision",
            "match_score",
            "fairness_mode",
        ]
    )

    for candidate in ranked_candidates:
        hire_prob = _safe_float(candidate.get("hire_probability", candidate.get("ranking_score", 0.0)))
        decision = candidate.get("decision")
        if not decision:
            decision = "HIRE" if hire_prob >= hire_threshold else "REJECT"

        writer.writerow(
            [
                candidate.get("resume_id", ""),
                candidate.get("candidate_name") or candidate.get("resume_data", {}).get("name", ""),
                candidate.get("rank", ""),
                round(hire_prob, 4),
                decision,
                candidate.get("match_score", ""),
                fairness_mode,
            ]
        )

    return output.getvalue().encode("utf-8")


def build_audit_pdf(
    job_id: str,
    ranked_candidates: List[Dict[str, Any]],
    fairness_metrics: Dict[str, Any],
) -> bytes:
    try:
        import fitz  # PyMuPDF uses the fitz namespace
        if not hasattr(fitz, "open"):
            raise ImportError("fitz module without open() detected")
    except Exception:
        try:
            import pymupdf as fitz  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyMuPDF is required for PDF export") from exc

    doc = fitz.open()
    page = doc.new_page()

    font_size = 11
    line_height = 14
    x = 50
    y = 60

    def add_line(text: str):
        nonlocal page, y
        if y > 780:
            page = doc.new_page()
            y = 60
        page.insert_text((x, y), text, fontsize=font_size)
        y += line_height

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    add_line("Fairness & Hallucination Audit Report")
    add_line(f"Job ID: {job_id}")
    add_line(f"Generated: {now}")
    add_line("")

    add_line("Fairness Metrics")
    add_line(f"Fairness Mode: {fairness_metrics.get('fairness_mode', 'OFF')}")
    add_line(f"Impact Ratio: {fairness_metrics.get('impact_ratio')}")
    add_line(f"Demographic Parity Diff: {fairness_metrics.get('demographic_parity')}")
    add_line(f"Equal Opportunity Diff: {fairness_metrics.get('equal_opportunity')}")
    add_line("")

    summary, flagged_lines = _verification_summary(ranked_candidates)
    add_line("Hallucination Verification Summary")
    add_line(f"Total Candidates: {summary['total_candidates']}")
    add_line(f"Flagged Candidates: {summary['flagged_candidates']}")
    add_line(f"Total Flagged Claims: {summary['total_flagged_claims']}")
    add_line(f"Verdicts: {summary['verdict_counts']}")

    if flagged_lines:
        add_line("")
        add_line("Flagged Claims (sample):")
        for line in flagged_lines[:10]:
            add_line(f"- {line}")

    return doc.tobytes()
