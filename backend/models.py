
from typing import List, Dict, Any, Optional

from pydantic import BaseModel

# Pydantic Models
class JDExtractionRequest(BaseModel):
    jd_text: str
    job_title: Optional[str] = None

class JDExtractionResponse(BaseModel):
    job_id: str
    job_title: Optional[str]
    required_skills: List[str]
    required_experience: int
    required_education: str
    certifications: List[str]
    status: str

class ResumeParseRequest(BaseModel):
    resume_text: str
    candidate_name: Optional[str] = None

class ResumeParseResponse(BaseModel):
    resume_id: str
    candidate_name: Optional[str]
    skills: List[str]
    education: List[Dict[str, str]]
    experience: Dict[str, Any]
    certifications: List[str]
    extraction_status: str

class VerifyClaimRequest(BaseModel):
    extraction: Any
    ground_truth: Optional[Any] = None

class VerifyResumeRequest(BaseModel):
    resume_id: str
    resume_extractions: Dict[str, Any]
    ground_truth_data: Optional[Dict[str, Any]] = None

class MatchRequest(BaseModel):
    resume_id: str
    job_id: str
    resume_data: Dict[str, Any]
    jd_data: Dict[str, Any]

class RankRequest(BaseModel):
    job_id: str
    candidates: List[Dict[str, Any]]
    jd_data: Dict[str, Any]
    use_fairness: bool = True
    sensitive_attribute: Optional[str] = "gender"
    hire_threshold: Optional[float] = 0.5
