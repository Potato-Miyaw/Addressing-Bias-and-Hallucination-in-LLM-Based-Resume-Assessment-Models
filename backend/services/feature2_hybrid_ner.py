"""
Feature 2C: Hybrid NER - Best of Both Worlds
Combines Generic BERT NER + Resume-Specific NER
Takes the best results from each model
"""

from typing import Dict, Any, List
import logging

from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.services.feature2_resume_ner_v2 import ResumeNERExtractorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridResumeNERExtractor:
    def __init__(self):
        """Initialize both NER models"""
        logger.info("Initializing Hybrid NER (Generic + Resume-Specific)")
        
        self.generic_ner = ResumeNERExtractor()
        self.resume_ner = ResumeNERExtractorV2()
        
        logger.info("✅ Hybrid NER ready")
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume using BOTH models and combine best results
        
        Strategy:
        - Name: Take first non-"Unknown" result
        - Skills: Combine both (more is better)
        - Experience: Take highest value
        - Education: Combine both
        - Companies: Combine both
        """
        if not resume_text:
            return self._empty_result("Empty resume text")
        
        try:
            logger.info("Running BOTH NER models...")
            
            # Run both models
            result_generic = self.generic_ner.parse_resume(resume_text)
            result_resume = self.resume_ner.parse_resume(resume_text, use_advanced=True)
            
            logger.info(f"Generic NER: {result_generic['extraction_status']}")
            logger.info(f"Resume NER: {result_resume['extraction_status']}")
            
            # Combine results intelligently
            combined = {}
            
            # NAME: Take first non-"Unknown"
            if result_generic['name'] != "Unknown":
                combined['name'] = result_generic['name']
            elif result_resume['name'] != "Unknown":
                combined['name'] = result_resume['name']
            else:
                combined['name'] = "Unknown"
            
            logger.info(f"Name: Generic='{result_generic['name']}', Resume='{result_resume['name']}' → Selected: '{combined['name']}'")
            
            # EMAIL: Take first non-"unknown@email.com"
            if result_generic['email_address'] != "unknown@email.com":
                combined['email_address'] = result_generic['email_address']
            elif result_resume['email_address'] != "unknown@email.com":
                combined['email_address'] = result_resume['email_address']
            else:
                combined['email_address'] = "unknown@email.com"
            
            # PHONE: Combine both
            phones_generic = result_generic.get('contact_number', [])
            phones_resume = result_resume.get('contact_number', [])
            combined['contact_number'] = list(set(phones_generic + phones_resume))[:3]
            
            # SKILLS: COMBINE BOTH (more is better!)
            skills_generic = set(result_generic.get('primary_skills', []))
            skills_resume = set(result_resume.get('primary_skills', []))
            all_skills = sorted(list(skills_generic | skills_resume))
            
            combined['primary_skills'] = all_skills[:20] if all_skills else []
            combined['secondary_skills'] = all_skills[20:40] if len(all_skills) > 20 else []
            
            logger.info(f"Skills: Generic={len(skills_generic)}, Resume={len(skills_resume)} → Combined: {len(all_skills)}")
            
            # EDUCATION: Combine both
            edu_generic = result_generic.get('education', [])
            edu_resume = result_resume.get('education', [])
            combined['education'] = list(set(edu_generic + edu_resume))[:5]
            
            # EXPERIENCE: Take MAXIMUM (more is better)
            exp_generic = result_generic.get('total_experience_(months)', 0)
            exp_resume = result_resume.get('total_experience_(months)', 0)
            combined['total_experience_(months)'] = max(exp_generic, exp_resume)
            
            logger.info(f"Experience: Generic={exp_generic}mo, Resume={exp_resume}mo → Selected: {combined['total_experience_(months)']}")
            
            # COMPANY: Take first non-"Unknown"
            if result_generic['current_company_name'] != "Unknown":
                combined['current_company_name'] = result_generic['current_company_name']
            elif result_resume['current_company_name'] != "Unknown":
                combined['current_company_name'] = result_resume['current_company_name']
            else:
                combined['current_company_name'] = "Unknown"
            
            # LOCATION: Take first non-"Unknown"
            if result_generic['current_location'] != "Unknown":
                combined['current_location'] = result_generic['current_location']
            elif result_resume['current_location'] != "Unknown":
                combined['current_location'] = result_resume['current_location']
            else:
                combined['current_location'] = "Unknown"
            
            # DESIGNATION: Combine both
            desig_generic = result_generic.get('designation', [])
            desig_resume = result_resume.get('designation', [])
            combined['designation'] = list(set(desig_generic + desig_resume))[:3]
            
            # OTHER FIELDS
            combined['relevant_experience_(primary)'] = result_generic.get('relevant_experience_(primary)', {"job_history": []})
            combined['relevant_experience_(secondary)'] = result_generic.get('relevant_experience_(secondary)', {"projects": [], "certifications": []})
            combined['applicant_description'] = resume_text[:200] + "..."
            
            # Combine entities from both
            entities_generic = result_generic.get('entities', [])
            entities_resume = result_resume.get('entities', [])
            combined['entities'] = entities_generic + entities_resume
            
            combined['extraction_status'] = "SUCCESS"
            combined['model_type'] = "HYBRID (Generic + Resume-Specific)"
            
            logger.info(f"✅ Hybrid extraction complete")
            
            return combined
        
        except Exception as e:
            logger.error(f"Hybrid parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))
    
    def simple_read(self, resume_text: str) -> Dict[str, Any]:
        """
        Lightweight read - use resume-specific NER for better name extraction
        """
        if not resume_text:
            return {"error": "Empty resume text", "status": "FAILED"}
        
        try:
            import hashlib
            resume_id = hashlib.md5(resume_text.encode()).hexdigest()[:12]
            
            # Try resume-specific NER first
            result_resume = self.resume_ner.simple_read(resume_text)
            name_resume = result_resume.get("name", "Unknown")
            
            # Fallback to generic NER if needed
            if name_resume == "Unknown":
                result_generic = self.generic_ner.simple_read(resume_text)
                name_resume = result_generic.get("name", "Unknown")
            
            # Get contact info from generic (more reliable)
            result_generic = self.generic_ner.simple_read(resume_text)
            
            return {
                "resume_id": resume_id,
                "name": name_resume,
                "email": result_generic.get("email", "unknown@email.com"),
                "phone": result_generic.get("phone", ""),
                "text": resume_text,
                "status": "STORED",
                "message": "Resume stored. Full extraction during matching."
            }
        except Exception as e:
            logger.error(f"Simple read failed: {e}")
            return {"error": str(e), "status": "FAILED"}
    
    def _empty_result(self, error_msg: str = "Unknown error") -> Dict[str, Any]:
        """Empty result on error"""
        return {
            "name": "Unknown",
            "designation": [],
            "contact_number": [],
            "email_address": "unknown@email.com",
            "education": [],
            "current_company_name": "Unknown",
            "current_location": "Unknown",
            "primary_skills": [],
            "secondary_skills": [],
            "total_experience_(months)": 0,
            "relevant_experience_(primary)": {"job_history": []},
            "relevant_experience_(secondary)": {"projects": [], "certifications": []},
            "applicant_description": "",
            "entities": [],
            "extraction_status": "FAILED",
            "error": error_msg
        }