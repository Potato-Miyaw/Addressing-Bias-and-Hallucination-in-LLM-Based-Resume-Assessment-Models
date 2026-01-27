"""
Audit report utilities: CSV shortlist + PDF fairness/hallucination summary.
"""

from io import BytesIO, StringIO
from typing import Dict, Any, List
import csv
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _verification_summary(candidates: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "verified": 0,
        "contains_hallucinations": 0,
        "no_claims": 0,
        "unknown": 0,
        "flagged_claims": 0
    }
    for candidate in candidates:
        verification = candidate.get("verification")
        status = candidate.get("verification_status")
        verdict = None
        if isinstance(verification, dict):
            verdict = verification.get("verdict")
        if verdict is None:
            verdict = status
        verdict = str(verdict or "UNKNOWN").upper()
        if "HALLUCINATION" in verdict:
            summary["contains_hallucinations"] += 1
        elif "NO_CLAIMS" in verdict:
            summary["no_claims"] += 1
        elif "VERIFIED" in verdict:
            summary["verified"] += 1
        else:
            summary["unknown"] += 1

        if isinstance(verification, dict):
            flagged = verification.get("flagged") or verification.get("flagged_claims") or []
            if isinstance(flagged, list):
                summary["flagged_claims"] += len(flagged)
    return summary


def build_shortlist_csv(
    job_id: str,
    ranked_candidates: List[Dict[str, Any]],
    fairness_mode: str,
    fairness_metrics: Dict[str, Any]
) -> bytes:
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(["job_id", job_id])
    writer.writerow(["fairness_mode", fairness_mode])
    writer.writerow([])

    writer.writerow([
        "resume_id",
        "rank",
        "hire_probability",
        "match_score",
        "decision",
        "verification_status"
    ])

    for candidate in ranked_candidates:
        writer.writerow([
            candidate.get("resume_id", ""),
            candidate.get("rank", ""),
            round(float(candidate.get("hire_probability", 0.0)), 4),
            candidate.get("match_score", ""),
            candidate.get("decision", ""),
            candidate.get("verification_status", "")
        ])

    writer.writerow([])
    writer.writerow(["impact_ratio", fairness_metrics.get("impact_ratio")])
    writer.writerow(["demographic_parity", fairness_metrics.get("demographic_parity")])
    writer.writerow(["equal_opportunity", fairness_metrics.get("equal_opportunity")])

    return output.getvalue().encode("utf-8")


def build_audit_pdf(
    job_id: str,
    ranked_candidates: List[Dict[str, Any]],
    fairness_mode: str,
    fairness_metrics: Dict[str, Any]
) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Fairness & Hallucination Audit Report")

    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Job ID: {job_id}")
    y -= 15
    c.drawString(50, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 15
    c.drawString(50, y, f"Fairness Mode: {fairness_mode}")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Fairness Metrics")

    y -= 15
    c.setFont("Helvetica", 10)
    def _fmt(value):
        return "N/A" if value is None else value

    c.drawString(60, y, f"Impact Ratio: {_fmt(fairness_metrics.get('impact_ratio'))}")
    y -= 12
    c.drawString(60, y, f"Demographic Parity Diff: {_fmt(fairness_metrics.get('demographic_parity'))}")
    y -= 12
    c.drawString(60, y, f"Equal Opportunity Diff: {_fmt(fairness_metrics.get('equal_opportunity'))}")
    y -= 12
    selection_rates = fairness_metrics.get("selection_rates")
    if selection_rates:
        c.drawString(60, y, f"Selection Rates: {selection_rates}")
    else:
        c.drawString(60, y, "Selection Rates: N/A")
    y -= 12
    if fairness_metrics.get("note"):
        c.drawString(60, y, f"Note: {fairness_metrics.get('note')}")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Verification Summary")
    y -= 15
    c.setFont("Helvetica", 10)
    summary = _verification_summary(ranked_candidates)
    c.drawString(60, y, f"Verified: {summary['verified']}")
    y -= 12
    c.drawString(60, y, f"Contains Hallucinations: {summary['contains_hallucinations']}")
    y -= 12
    c.drawString(60, y, f"No Claims: {summary['no_claims']}")
    y -= 12
    c.drawString(60, y, f"Unknown: {summary['unknown']}")
    y -= 12
    c.drawString(60, y, f"Flagged Claims (total): {summary['flagged_claims']}")
    if summary["unknown"] == len(ranked_candidates):
        y -= 12
        c.drawString(60, y, "Note: Verification data not provided for these candidates.")

    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Top Candidates")
    y -= 15
    c.setFont("Helvetica-Bold", 9)
    c.drawString(60, y, "Rank | Resume ID | Name | Hire Prob | Match | Decision | Verify")
    y -= 12
    c.setFont("Helvetica", 9)

    for candidate in ranked_candidates[:10]:
        resume_id = candidate.get("resume_id", "")
        name = candidate.get("candidate_name")
        if not name and isinstance(candidate.get("resume_data"), dict):
            name = candidate["resume_data"].get("name")
        name = name or "Unknown"
        hire_prob = round(float(candidate.get("hire_probability", 0.0)), 3)
        match_score = candidate.get("match_score")
        if match_score is None and isinstance(candidate.get("match_data"), dict):
            match_score = candidate["match_data"].get("match_score")
        if match_score is None:
            match_score = "N/A"
        else:
            match_val = float(match_score)
            match_score = f"{match_val:.2f}" if match_val <= 1 else f"{match_val:.1f}%"

        line = (
            f"{candidate.get('rank', '')} | "
            f"{resume_id} | "
            f"{name} | "
            f"{hire_prob} | "
            f"{match_score} | "
            f"{candidate.get('decision', '')} | "
            f"{candidate.get('verification_status', '')}"
        )
        c.drawString(60, y, line)
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
