"""
Audit Report Router
Exports shortlist CSV and fairness/hallucination PDF as a zip archive.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from io import BytesIO
import zipfile

from backend.utils.reporting import build_shortlist_csv, build_audit_pdf

router = APIRouter(prefix="/api/audit", tags=["Audit"])


class AuditExportRequest(BaseModel):
    job_id: str
    ranked_candidates: List[Dict[str, Any]]
    fairness_metrics: Optional[Dict[str, Any]] = None
    fairness_mode: Optional[str] = "OFF"


@router.post("/export")
async def export_audit_report(request: AuditExportRequest):
    try:
        fairness_metrics = request.fairness_metrics or {}
        fairness_mode = request.fairness_mode or "OFF"

        csv_bytes = build_shortlist_csv(
            job_id=request.job_id,
            ranked_candidates=request.ranked_candidates,
            fairness_mode=fairness_mode,
            fairness_metrics=fairness_metrics
        )
        pdf_bytes = build_audit_pdf(
            job_id=request.job_id,
            ranked_candidates=request.ranked_candidates,
            fairness_mode=fairness_mode,
            fairness_metrics=fairness_metrics
        )

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"{request.job_id}_shortlist.csv", csv_bytes)
            zip_file.writestr(f"{request.job_id}_audit_report.pdf", pdf_bytes)

        zip_buffer.seek(0)
        filename = f"{request.job_id}_audit_bundle.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit export failed: {str(e)}")
