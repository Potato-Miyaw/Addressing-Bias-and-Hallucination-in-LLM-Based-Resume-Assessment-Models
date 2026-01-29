"""
Resume Parsing Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import hashlib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.services.feature2_resume_ner_v2 import ResumeNERExtractorV2
from backend.services.feature2_hybrid_ner import HybridResumeNERExtractor
from backend.utils.document_parser import extract_text_from_file
from backend.utils.id_generator import generate_resume_id
from backend import database

router = APIRouter(prefix="/api/resume", tags=["Resume"])

# Lazy load services
ner_extractor = None
ner_extractor_v2 = None
hybrid_extractor = None

def get_ner_extractor():
    global ner_extractor
    if ner_extractor is None:
        ner_extractor = ResumeNERExtractor()
    return ner_extractor

def get_ner_extractor_v2():
    global ner_extractor_v2
    if ner_extractor_v2 is None:
        ner_extractor_v2 = ResumeNERExtractorV2()
    return ner_extractor_v2

def get_hybrid_extractor():
    global hybrid_extractor
    if hybrid_extractor is None:
        hybrid_extractor = HybridResumeNERExtractor()
    return hybrid_extractor

# Pydantic models
class ResumeParseRequest(BaseModel):
    resume_text: str
    candidate_name: Optional[str] = None
    save_to_db: bool = True
    upload_source: str = "candidate_self"
    uploaded_by: str = "self"

@router.post("/parse")
async def parse_resume(request: ResumeParseRequest):
    """
    Full resume parsing with Generic BERT NER - extracts all fields including skills, experience, education
    Saves raw text to ground_truth collection if save_to_db=True
    """
    try:
        # Generate IDs
        resume_id, content_hash = generate_resume_id(request.resume_text, request.candidate_name or "candidate")
        
        # Check for duplicates
        is_duplicate = False
        existing_resume = None
        if request.save_to_db:
            existing_resume = await database.get_ground_truth_by_content_hash(content_hash)
            if existing_resume:
                is_duplicate = True
                resume_id = existing_resume["resume_id"]
        
        # Save ground truth (raw text)
        saved_to_db = False
        if request.save_to_db and not is_duplicate:
            ground_truth_data = {
                "resume_id": resume_id,
                "content_hash": content_hash,
                "raw_text": request.resume_text,
                "filename": "text_input.txt",
                "file_type": "TXT",
                "text_length": len(request.resume_text),
                "candidate_name": request.candidate_name,
                "candidate_email": None,  # Will be extracted after NER
                "upload_source": request.upload_source,
                "uploaded_by": request.uploaded_by
            }
            saved_to_db = await database.save_ground_truth(ground_truth_data)
        
        # Run NER extraction
        extractor = get_ner_extractor()
        resume_data = extractor.parse_resume(request.resume_text)
        
        if resume_data.get("status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume parsing failed")
            )
        
        # Add candidate name if provided
        if request.candidate_name:
            resume_data["candidate_name"] = request.candidate_name
        
        # Extract certifications from nested structure
        certifications = []
        relevant_exp_secondary = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(relevant_exp_secondary, dict):
            certifications = relevant_exp_secondary.get("certifications", [])
        
        # Extract complete job history
        job_history = []
        relevant_exp_primary = resume_data.get("relevant_experience_(primary)", {})
        if isinstance(relevant_exp_primary, dict):
            job_history = relevant_exp_primary.get("job_history", [])
        
        # Build company list
        companies = resume_data.get("companies", [])
        if not companies and job_history:
            for job in job_history:
                if isinstance(job, dict) and job.get("company"):
                    companies.append(job["company"])
                elif isinstance(job, str):
                    companies.append(job)
        
        response = {
            "success": True,
            "resume_id": resume_id,
            "name": resume_data.get("name"),
            "email": resume_data.get("email_address", resume_data.get("email")),
            "phone": resume_data.get("phone"),
            "location": resume_data.get("current_location", "Unknown"),
            "skills": resume_data.get("skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(years)", 0),
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": companies,
                "job_history": job_history
            },
            "education": resume_data.get("education", []),
            "certifications": certifications,
            "status": resume_data.get("status"),
            "message": resume_data.get("message"),
            "saved_to_db": saved_to_db,
            "model_type": "Generic BERT NER"
        }
        
        # Add duplicate warning if applicable
        if is_duplicate and existing_resume:
            response["duplicate_warning"] = f"Resume with same content already exists (ID: {existing_resume['resume_id']})"
            response["is_duplicate"] = True
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume reading failed: {str(e)}"
        )

@router.post("/parse-v2")
async def parse_resume_v2(request: ResumeParseRequest):
    """
    Full resume parsing with Resume-Specific BERT NER V2 (yashpwr/resume-ner-bert-v2)
    """
    try:
        extractor = get_ner_extractor_v2()
        resume_data = extractor.parse_resume(request.resume_text, use_advanced=True)
        
        if resume_data.get("extraction_status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume parsing failed")
            )
        
        # Add candidate name if provided
        if request.candidate_name:
            resume_data["candidate_name"] = request.candidate_name
        
        # Extract certifications from nested structure
        certifications = []
        relevant_exp_secondary = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(relevant_exp_secondary, dict):
            certifications = relevant_exp_secondary.get("certifications", [])
        
        # Extract complete job history
        job_history = []
        relevant_exp_primary = resume_data.get("relevant_experience_(primary)", {})
        if isinstance(relevant_exp_primary, dict):
            job_history = relevant_exp_primary.get("job_history", [])
        
        # Build company list from job history or fallback to current company
        companies = []
        if job_history:
            for job in job_history:
                if isinstance(job, dict) and job.get("company"):
                    companies.append(job["company"])
                elif isinstance(job, str):
                    companies.append(job)
        
        if not companies:
            current_company = resume_data.get("current_company_name", "Unknown")
            if current_company and current_company != "Unknown":
                companies = [current_company]
        
        # Build response matching test expectations
        return {
            "success": True,
            "resume_id": hashlib.md5(request.resume_text.encode()).hexdigest()[:12],
            "name": resume_data.get("name", "Unknown"),
            "email": resume_data.get("email_address", "unknown@email.com"),
            "phone": resume_data.get("contact_number", [""])[0] if resume_data.get("contact_number") else "",
            "location": resume_data.get("current_location", "Unknown"),
            "skills": resume_data.get("skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(months)", 0) // 12,
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": companies,
                "job_history": job_history
            },
            "education": resume_data.get("education", []),
            "certifications": certifications,
            "status": resume_data.get("extraction_status", "SUCCESS"),
            "message": f"Parsed with {resume_data.get('model_type', 'Resume-NER-V2')}",
            "saved_to_db": False,
            "model_type": resume_data.get("model_type", "Resume-NER-V2"),
            "entities": resume_data.get("entities", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume V2 parsing failed: {str(e)}"
        )

@router.post("/parse-hybrid")
async def parse_resume_hybrid(
    file: UploadFile = File(...), 
    candidate_name: Optional[str] = Form(None), 
    save_to_db: bool = Form(True),
    upload_source: str = Form("candidate_self"),
    uploaded_by: str = Form("self")
):
    """
    HYBRID Resume Parsing - Best of Both Worlds!
    Combines Generic BERT NER (good at names) + Resume-Specific NER (good at skills)
    Accepts file upload (PDF, DOCX, DOC, TXT)
    Saves raw text to ground_truth collection if save_to_db=True
    
    Args:
        upload_source: "candidate_self" (default) or "hr_upload"
        uploaded_by: "self" (default) or HR user ID
    """
    try:
        # Debug logging
        print(f"🔍 DEBUG parse-hybrid - upload_source: {upload_source}, uploaded_by: {uploaded_by}")
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text or not resume_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        # Generate IDs
        resume_id, content_hash = generate_resume_id(resume_text, candidate_name or "candidate")
        
        # Check for duplicates
        is_duplicate = False
        existing_resume = None
        if save_to_db:
            existing_resume = await database.get_ground_truth_by_content_hash(content_hash)
            if existing_resume:
                is_duplicate = True
                resume_id = existing_resume["resume_id"]
        
        # Run NER extraction first to get email
        extractor = get_hybrid_extractor()
        resume_data = extractor.parse_resume(resume_text)
        
        # Extract email from NER results
        extracted_email = resume_data.get("email_address")
        if extracted_email == "unknown@email.com":
            extracted_email = None
        
        # Save ground truth (raw text)
        saved_to_db = False
        if save_to_db and not is_duplicate:
            ground_truth_data = {
                "resume_id": resume_id,
                "content_hash": content_hash,
                "raw_text": resume_text,
                "filename": file.filename,
                "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
                "text_length": len(resume_text),
                "candidate_name": candidate_name or resume_data.get("name", "Unknown"),
                "candidate_email": extracted_email,
                "upload_source": upload_source,
                "uploaded_by": uploaded_by
            }
            saved_to_db = await database.save_ground_truth(ground_truth_data)
        
        if resume_data.get("extraction_status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Hybrid resume parsing failed")
            )
        
        if candidate_name:
            resume_data["candidate_name"] = candidate_name
        
        # Extract certifications from nested structure
        certifications = []
        relevant_exp_secondary = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(relevant_exp_secondary, dict):
            certifications = relevant_exp_secondary.get("certifications", [])
        
        # Extract complete job history
        job_history = []
        relevant_exp_primary = resume_data.get("relevant_experience_(primary)", {})
        if isinstance(relevant_exp_primary, dict):
            job_history = relevant_exp_primary.get("job_history", [])
        
        # Build company list from job history or fallback to current company
        companies = []
        if job_history:
            for job in job_history:
                if isinstance(job, dict) and job.get("company"):
                    companies.append(job["company"])
                elif isinstance(job, str):
                    companies.append(job)
        
        # Fallback to current company if no history
        if not companies:
            current_company = resume_data.get("current_company_name", "Unknown")
            if current_company and current_company != "Unknown":
                companies = [current_company]
        
        response = {
            "success": True,
            "resume_id": resume_id,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
            "text": resume_text,
            "text_length": len(resume_text),
            "name": resume_data.get("name", "Unknown"),
            "email": resume_data.get("email_address", "unknown@email.com"),
            "phone": resume_data.get("contact_number", [""])[0] if resume_data.get("contact_number") else "",
            "location": resume_data.get("current_location", "Unknown"),
            "skills": resume_data.get("skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(months)", 0) // 12,
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": companies,
                "job_history": job_history
            },
            "education": resume_data.get("education", []),
            "certifications": certifications,
            "status": resume_data.get("extraction_status", "SUCCESS"),
            "message": f"Parsed with {resume_data.get('model_type', 'HYBRID')}",
            "saved_to_db": saved_to_db,
            "model_type": resume_data.get("model_type", "HYBRID (Generic + Resume-Specific)"),
            "entities": resume_data.get("entities", [])
        }
        
        # Add duplicate warning if applicable
        if is_duplicate and existing_resume:
            response["duplicate_warning"] = f"Resume with same content already exists (ID: {existing_resume['resume_id']})"
            response["is_duplicate"] = True
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid resume parsing failed: {str(e)}"
        )

@router.post("/parse-generic-file")
async def parse_resume_generic_file(
    file: UploadFile = File(...), 
    candidate_name: Optional[str] = Form(None), 
    save_to_db: bool = Form(True),
    upload_source: str = Form("candidate_self"),
    uploaded_by: str = Form("self")
):
    """
    Generic BERT NER with file upload (PDF, DOCX, DOC, TXT)
    Saves raw text to ground_truth collection if save_to_db=True
    
    Args:
        upload_source: "candidate_self" (default) or "hr_upload"
        uploaded_by: "self" (default) or HR user ID
    """
    try:
        # Debug logging
        print(f"🔍 DEBUG - upload_source received: {upload_source}")
        print(f"🔍 DEBUG - uploaded_by received: {uploaded_by}")
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text or not resume_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        # Generate IDs
        resume_id, content_hash = generate_resume_id(resume_text, candidate_name or "candidate")
        
        # Check for duplicates
        is_duplicate = False
        existing_resume = None
        if save_to_db:
            existing_resume = await database.get_ground_truth_by_content_hash(content_hash)
            if existing_resume:
                is_duplicate = True
                resume_id = existing_resume["resume_id"]
        
        # Run NER extraction first to get email
        extractor = get_ner_extractor()
        resume_data = extractor.parse_resume(resume_text)
        
        # Extract email from NER results
        extracted_email = resume_data.get("email_address", resume_data.get("email"))
        if extracted_email == "unknown@email.com":
            extracted_email = None
        
        # Save ground truth (raw text)
        saved_to_db = False
        if save_to_db and not is_duplicate:
            ground_truth_data = {
                "resume_id": resume_id,
                "content_hash": content_hash,
                "raw_text": resume_text,
                "filename": file.filename,
                "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
                "text_length": len(resume_text),
                "candidate_name": candidate_name or resume_data.get("name", "Unknown"),
                "candidate_email": extracted_email,
                "upload_source": upload_source,
                "uploaded_by": uploaded_by
            }
            saved_to_db = await database.save_ground_truth(ground_truth_data)
        
        if resume_data.get("status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Generic NER parsing failed")
            )
        
        if candidate_name:
            resume_data["candidate_name"] = candidate_name
        
        # Extract certifications from nested structure
        certifications = []
        relevant_exp_secondary = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(relevant_exp_secondary, dict):
            certifications = relevant_exp_secondary.get("certifications", [])
        
        # Extract complete job history
        job_history = []
        relevant_exp_primary = resume_data.get("relevant_experience_(primary)", {})
        if isinstance(relevant_exp_primary, dict):
            job_history = relevant_exp_primary.get("job_history", [])
        
        # Build company list
        companies = resume_data.get("companies", [])
        if not companies and job_history:
            for job in job_history:
                if isinstance(job, dict) and job.get("company"):
                    companies.append(job["company"])
                elif isinstance(job, str):
                    companies.append(job)
        
        response = {
            "success": True,
            "resume_id": resume_id,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
            "text": resume_text,
            "text_length": len(resume_text),
            "name": resume_data.get("name", "Unknown"),
            "email": resume_data.get("email_address", resume_data.get("email", "unknown@email.com")),
            "phone": resume_data.get("phone", ""),
            "location": resume_data.get("current_location", "Unknown"),
            "skills": resume_data.get("skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(years)", 0),
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": companies,
                "job_history": job_history
            },
            "education": resume_data.get("education", []),
            "certifications": certifications,
            "status": resume_data.get("status", "SUCCESS"),
            "message": resume_data.get("message", "Parsed with Generic BERT NER"),
            "saved_to_db": saved_to_db,
            "model_type": "Generic BERT NER"
        }
        
        # Add duplicate warning if applicable
        if is_duplicate and existing_resume:
            response["duplicate_warning"] = f"Resume with same content already exists (ID: {existing_resume['resume_id']})"
            response["is_duplicate"] = True
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generic NER file parsing failed: {str(e)}"
        )

@router.post("/parse-v2-file")
async def parse_resume_v2_file(
    file: UploadFile = File(...), 
    candidate_name: Optional[str] = Form(None), 
    save_to_db: bool = Form(True),
    upload_source: str = Form("candidate_self"),
    uploaded_by: str = Form("self")
):
    """
    Resume-Specific BERT NER V2 with file upload (PDF, DOCX, DOC, TXT)
    Saves raw text to ground_truth collection if save_to_db=True
    
    Args:
        upload_source: "candidate_self" (default) or "hr_upload"
        uploaded_by: "self" (default) or HR user ID
    """
    try:
        # Debug logging
        print(f"🔍 DEBUG parse-v2-file - upload_source: {upload_source}, uploaded_by: {uploaded_by}")
        
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text or not resume_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        # Generate IDs
        resume_id, content_hash = generate_resume_id(resume_text, candidate_name or "candidate")
        
        # Check for duplicates
        is_duplicate = False
        existing_resume = None
        if save_to_db:
            existing_resume = await database.get_ground_truth_by_content_hash(content_hash)
            if existing_resume:
                is_duplicate = True
                resume_id = existing_resume["resume_id"]
        
        # Run NER extraction first to get email
        extractor = get_ner_extractor_v2()
        resume_data = extractor.parse_resume(resume_text, use_advanced=True)
        
        # Extract email from NER results
        extracted_email = resume_data.get("email_address")
        if extracted_email == "unknown@email.com":
            extracted_email = None
        
        # Save ground truth (raw text)
        saved_to_db = False
        if save_to_db and not is_duplicate:
            ground_truth_data = {
                "resume_id": resume_id,
                "content_hash": content_hash,
                "raw_text": resume_text,
                "filename": file.filename,
                "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
                "text_length": len(resume_text),
                "candidate_name": candidate_name or resume_data.get("name", "Unknown"),
                "candidate_email": extracted_email,
                "upload_source": upload_source,
                "uploaded_by": uploaded_by
            }
            saved_to_db = await database.save_ground_truth(ground_truth_data)
        
        if resume_data.get("extraction_status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume-Specific NER parsing failed")
            )
        
        if candidate_name:
            resume_data["candidate_name"] = candidate_name
        
        # Extract certifications from nested structure
        certifications = []
        relevant_exp_secondary = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(relevant_exp_secondary, dict):
            certifications = relevant_exp_secondary.get("certifications", [])
        
        # Extract complete job history
        job_history = []
        relevant_exp_primary = resume_data.get("relevant_experience_(primary)", {})
        if isinstance(relevant_exp_primary, dict):
            job_history = relevant_exp_primary.get("job_history", [])
        
        # Build company list from job history or fallback to current company
        companies = []
        if job_history:
            for job in job_history:
                if isinstance(job, dict) and job.get("company"):
                    companies.append(job["company"])
                elif isinstance(job, str):
                    companies.append(job)
        
        if not companies:
            current_company = resume_data.get("current_company_name", "Unknown")
            if current_company and current_company != "Unknown":
                companies = [current_company]
        
        response = {
            "success": True,
            "resume_id": resume_id,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper() if '.' in file.filename else "UNKNOWN",
            "text": resume_text,
            "text_length": len(resume_text),
            "name": resume_data.get("name", "Unknown"),
            "email": resume_data.get("email_address", "unknown@email.com"),
            "phone": resume_data.get("contact_number", [""])[0] if resume_data.get("contact_number") else "",
            "location": resume_data.get("current_location", "Unknown"),
            "skills": resume_data.get("skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(months)", 0) // 12,
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": companies,
                "job_history": job_history
            },
            "education": resume_data.get("education", []),
            "certifications": certifications,
            "status": resume_data.get("extraction_status", "SUCCESS"),
            "message": f"Parsed with {resume_data.get('model_type', 'Resume-NER-V2')}",
            "saved_to_db": saved_to_db,
            "model_type": resume_data.get("model_type", "Resume-NER-V2"),
            "entities": resume_data.get("entities", [])
        }
        
        # Add duplicate warning if applicable
        if is_duplicate and existing_resume:
            response["duplicate_warning"] = f"Resume with same content already exists (ID: {existing_resume['resume_id']})"
            response["is_duplicate"] = True
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume-Specific NER file parsing failed: {str(e)}"
        )

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...), save_to_db: bool = True):
    """
    Upload resume as file - supports PDF, DOCX, DOC, TXT
    Just stores text, NER extraction happens during matching
    """
    try:
        content = await file.read()
        
        # Extract text based on file type
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text or not resume_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        extractor = get_ner_extractor()
        resume_data = extractor.simple_read(resume_text)
        resume_data["filename"] = file.filename
        
        return {
            "resume_id": resume_data["resume_id"],
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper(),
            "text_length": len(resume_text),
            "name": resume_data["name"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "text": resume_data["text"],
            "status": resume_data["status"],
            "message": resume_data["message"],
            "saved_to_db": False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume upload failed: {str(e)}"
        )

@router.post("/batch")
async def batch_parse_resumes(files: List[UploadFile] = File(...), uploaded_by: str = "hr_batch"):
    """
    Upload multiple resumes - supports PDF, DOCX, DOC, TXT
    Saves to ground_truth collection with upload_source="hr_upload"
    """
    from backend.utils.id_generator import generate_resume_id, generate_content_hash
    from backend import database
    
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            # Extract text based on file type
            try:
                resume_text = extract_text_from_file(content, file.filename)
            except HTTPException as e:
                results.append({
                    "filename": file.filename,
                    "status": "FAILED",
                    "error": e.detail
                })
                continue
            
            if not resume_text or not resume_text.strip():
                results.append({
                    "filename": file.filename,
                    "status": "FAILED",
                    "error": "No text could be extracted"
                })
                continue
            
            # Run NER to extract candidate info
            extractor = get_ner_extractor()
            resume_data = extractor.parse_resume(resume_text)
            
            # Generate IDs
            candidate_name = resume_data.get('name', 'Unknown')
            resume_id = generate_resume_id(resume_text, candidate_name)
            content_hash = generate_content_hash(resume_text)
            
            # Check for duplicates
            existing = await database.get_ground_truth_by_content_hash(content_hash)
            is_duplicate = existing is not None
            
            # Extract candidate email from NER results
            candidate_email = resume_data.get('email')
            if not candidate_email:
                # Try to find in contact info
                contact_info = resume_data.get('contact_info', {})
                candidate_email = contact_info.get('email')
            
            saved_to_db = False
            
            if not is_duplicate:
                # Save to ground_truth collection
                ground_truth_entry = {
                    "resume_id": resume_id,
                    "content_hash": content_hash,
                    "raw_text": resume_text,
                    "filename": file.filename,
                    "file_type": file.filename.split('.')[-1].upper(),
                    "text_length": len(resume_text),
                    "candidate_name": candidate_name,
                    "candidate_email": candidate_email,
                    "upload_source": "hr_upload",
                    "uploaded_by": uploaded_by,
                    "uploaded_at": datetime.utcnow().isoformat()
                }
                
                saved_resume_id = await database.save_ground_truth(ground_truth_entry)
                saved_to_db = bool(saved_resume_id)
            
            results.append({
                "resume_id": resume_id,
                "filename": file.filename,
                "file_type": file.filename.split('.')[-1].upper(),
                "text_length": len(resume_text),
                "name": candidate_name,
                "email": candidate_email or "Not extracted",
                "phone": resume_data.get('phone') or resume_data.get('contact_info', {}).get('phone'),
                "status": "SUCCESS",
                "saved_to_db": saved_to_db,
                "is_duplicate": is_duplicate,
                "message": "Duplicate found - not saved" if is_duplicate else "Saved to database successfully"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "FAILED",
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": len([r for r in results if r["status"] == "SUCCESS"]),
        "failed": len([r for r in results if r["status"] == "FAILED"]),
        "saved": len([r for r in results if r.get("saved_to_db", False)]),
        "duplicates": len([r for r in results if r.get("is_duplicate", False)]),
        "results": results
    }


# ============== NER VISUALIZATION ENDPOINTS ==============

class NERVisualizationRequest(BaseModel):
    resume_text: str
    resume_data: Dict[str, Any]

@router.post("/ner/visualize")
async def visualize_ner_output(request: NERVisualizationRequest):
    """
    Get NER visualization data - entity table, highlighted text, and summary
    
    Returns:
    - entity_table: Formatted table of extracted entities with categories
    - entity_summary: Statistics about extracted entities
    - highlighted_text: HTML with highlighted entities in original text
    - extraction_status: Quality of extraction
    """
    try:
        extractor = get_ner_extractor()
        
        # Get visualization data
        viz_data = extractor.get_visualization_data(
            request.resume_data,
            request.resume_text
        )
        
        return {
            "success": True,
            "visualization": viz_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visualization failed: {str(e)}"
        )


@router.post("/ner/entity-table")
async def get_entity_table(request: NERVisualizationRequest):
    """
    Get formatted entity table from extracted resume data
    
    Returns table with columns: Category, Value, Confidence
    """
    try:
        extractor = get_ner_extractor()
        entity_table = extractor.get_entity_table(request.resume_data)
        
        return {
            "success": True,
            "total_entities": len(entity_table),
            "entity_table": entity_table
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity table generation failed: {str(e)}"
        )


@router.post("/ner/summary")
async def get_entity_summary(request: NERVisualizationRequest):
    """
    Get summary statistics of extracted entities
    
    Returns counts for:
    - Skills (primary & secondary)
    - Education degrees
    - Job titles
    - Certifications
    - Contact info
    """
    try:
        extractor = get_ner_extractor()
        summary = extractor.get_entity_summary(request.resume_data)
        
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summary generation failed: {str(e)}"
        )


@router.post("/ner/highlighted-text")
async def get_highlighted_text(request: NERVisualizationRequest):
    """
    Get HTML with highlighted entities in the resume text
    
    Different entity types are highlighted with different colors:
    - Skills: Gold
    - Education: Sky Blue
    - Job Titles: Light Green
    - Company: Plum
    - Certifications: Khaki
    """
    try:
        extractor = get_ner_extractor()
        highlighted = extractor.highlight_entities_in_text(
            request.resume_text,
            request.resume_data
        )
        
        return {
            "success": True,
            "highlighted_html": highlighted
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text highlighting failed: {str(e)}"
        )

@router.post("/extract-skills")
async def extract_skills(request: ResumeParseRequest):
    """
    Extract technical skills from resume text
    
    Returns a list of all identified technical skills including:
    - Programming languages (Python, Java, etc.)
    - Web frameworks (React, Django, etc.)
    - Databases (SQL, MongoDB, etc.)
    - Cloud platforms (AWS, Azure, etc.)
    - BI/Data tools (Power BI, Tableau, Excel, etc.)
    - And 20+ other skill categories
    """
    try:
        if not request.resume_text or not request.resume_text.strip():
            return {
                "success": True,
                "skills": [],
                "skill_count": 0,
                "message": "Empty resume text"
            }
        
        extractor = get_ner_extractor()
        skills = extractor.extract_skills(request.resume_text)
        
        return {
            "success": True,
            "skills": skills,
            "skill_count": len(skills),
            "message": f"Successfully extracted {len(skills)} skills"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill extraction failed: {str(e)}"
        )