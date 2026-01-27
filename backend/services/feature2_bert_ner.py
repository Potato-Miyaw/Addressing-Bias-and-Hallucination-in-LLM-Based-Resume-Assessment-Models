"""
Feature 2: BERT NER for Resume Parsing
Simple, fast, effective extraction
"""

from transformers import pipeline
from typing import List, Dict, Any
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeNERExtractor:
    def __init__(self, model_path: str = None):
        """Initialize BERT NER - simple and fast"""
        self.ner_pipeline = None
        
        try:
            logger.info("Loading BERT NER model: dslim/bert-base-NER")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1
            )
            logger.info("BERT NER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            self.ner_pipeline = None
    
    def extract_entities_bert(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using BERT NER"""
        if not self.ner_pipeline or not text:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            formatted_entities = []
            for entity in entities:
                entity_type = entity.get("entity_group", entity.get("entity", "O"))
                
                formatted_entities.append({
                    "entity_type": entity_type,
                    "value": entity["word"].strip().replace("##", ""),
                    "score": float(entity["score"]),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0)
                })
            
            return formatted_entities
        except Exception as e:
            logger.error(f"BERT NER extraction failed: {e}")
            return []
    
    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract phone and email"""
        contact = {"phone": [], "email": None}
        
        # Phone
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{10}',
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            contact["phone"].extend([m.strip() for m in matches])
        
        contact["phone"] = list(dict.fromkeys(contact["phone"]))[:3]
        
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact["email"] = email_match.group(0)
        
        return contact
    
    def extract_name(self, text: str) -> str:
        """Extract name - SIMPLE approach"""
        lines = text.split('\n')[:5]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 4:
                continue
            
            # Skip obvious non-name lines
            if any(word in line.lower() for word in ['resume', 'email', '@', 'phone', 'http', 'linkedin']):
                continue
            
            if re.search(r'\d{3,}', line):  # Skip lines with 3+ digits
                continue
            
            # Name patterns
            name_patterns = [
                r'^([A-Z][A-Z\s]+)$',  # ALL CAPS
                r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',  # Title Case
                r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+[A-Z][a-z]+)$',  # With middle initial
            ]
            
            for pattern in name_patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1).strip()
                    words = name.split()
                    if 2 <= len(words) <= 4 and 4 <= len(name) <= 60:
                        return name
        
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills - FOCUSED patterns"""
        skills = set()
        
        # Core tech skills only
        skill_patterns = [
            # Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|Scala)\b',
            
            # Web Frameworks
            r'\b(React|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Node\.js|Laravel|Rails|Next\.js)\b',
            
            # ML/AI
            r'\b(PyTorch|TensorFlow|Keras|scikit-learn|XGBoost|BERT|GPT|Transformers|Hugging\s*Face|spaCy|NLTK)\b',
            
            # Cloud
            r'\b(AWS|Azure|GCP|Google\s*Cloud|Heroku|DigitalOcean)\b',
            
            # DevOps
            r'\b(Docker|Kubernetes|Jenkins|GitLab|GitHub\s*Actions|Terraform|Ansible)\b',
            
            # Databases
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|DynamoDB|Snowflake)\b',
            
            # Data
            r'\b(Spark|Hadoop|Kafka|Airflow|Pandas|NumPy)\b',
            
            # Frontend
            r'\b(HTML|CSS|SASS|Bootstrap|Tailwind|jQuery|Webpack)\b',
            
            # Testing
            r'\b(pytest|Jest|Selenium|Cypress|JUnit)\b',
            
            # Tools
            r'\b(Git|Jira|VS\s*Code|Postman)\b',
            
            # Methods
            r'\b(Agile|Scrum|CI/CD|REST|GraphQL|Microservices)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        return sorted(list(skills))
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education"""
        education = []
        
        degree_patterns = [
            r'\b(PhD|Ph\.D|Doctor\s+of\s+Philosophy)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
            r'\b(Master|MS|MSc|M\.S|MA|MBA)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
            r'\b(Bachelor|BS|BSc|B\.S|BA|BTech)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern in degree_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    degree = match.group(1)
                    field = match.group(2).strip() if len(match.groups()) > 1 else "Unknown"
                    
                    # Get nearby context for institution
                    context = ' '.join(lines[max(0, i-1):min(len(lines), i+3)])
                    entities = self.extract_entities_bert(context)
                    
                    institutions = [
                        e["value"].replace("##", "")
                        for e in entities
                        if e["entity_type"] == "ORGANIZATION" and e["score"] > 0.5
                    ]
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', context)
                    
                    education.append({
                        "degree": f"{degree} in {field}",
                        "institution": institutions[0] if institutions else "Unknown",
                        "year": year_match.group(0) if year_match else "Unknown"
                    })
                    break
        
        return education
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience years and companies"""
        # Years
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:?\s*(\d+)\+?\s*years?',
        ]
        
        years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                break
        
        # Companies from BERT
        entities = self.extract_entities_bert(text)
        companies = [
            e["value"].replace("##", "")
            for e in entities
            if e["entity_type"] == "ORGANIZATION" and e["score"] > 0.6
        ]
        
        # Filter out universities
        companies = [c for c in companies if not any(word in c.lower() for word in ['university', 'college', 'institute'])]
        companies = list(dict.fromkeys(companies))[:5]
        
        # Job titles
        title_patterns = [
            r'\b(Senior|Lead|Principal)\s+(Engineer|Developer|Scientist)',
            r'\b(Software|Machine\s+Learning|Data)\s+(Engineer|Developer)',
        ]
        
        roles = set()
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    roles.add(' '.join(match))
        
        return {
            "years": years,
            "companies": companies,
            "roles": list(roles)[:5]
        }
    
    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications"""
        certs = []
        
        cert_patterns = [
            r'(AWS\s+Certified[^.\n]+)',
            r'(Azure[^.\n]+Certified[^.\n]+)',
            r'(Google\s+Cloud[^.\n]+Certified[^.\n]+)',
            r'\b(PMP|CISSP|CEH|CISA)\b',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cert = match.strip() if isinstance(match, str) else match[0].strip()
                if len(cert) > 5:
                    certs.append(cert)
        
        return list(dict.fromkeys(certs))
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Complete parsing - SIMPLE and FAST"""
        if not resume_text:
            return self._empty_result("Empty resume text")
        
        try:
            # Extract all fields
            name = self.extract_name(resume_text)
            contact_info = self.extract_contact_info(resume_text)
            skills = self.extract_skills(resume_text)
            education = self.extract_education(resume_text)
            experience = self.extract_experience(resume_text)
            certs = self.extract_certifications(resume_text)
            
            return {
                "name": name,
                "email_address": contact_info.get("email", "unknown@email.com"),
                "contact_number": contact_info["phone"],
                "primary_skills": skills,
                "secondary_skills": [],
                "education": [e["degree"] for e in education],
                "total_experience_(months)": experience["years"] * 12,
                "current_company_name": experience["companies"][0] if experience["companies"] else "Unknown",
                "current_location": "Unknown",
                "designation": experience["roles"],
                "relevant_experience_(primary)": {"job_history": []},
                "relevant_experience_(secondary)": {"projects": [], "certifications": []},
                "applicant_description": resume_text[:200] + "...",
                "entities": [],
                "extraction_status": "SUCCESS",
                "model_type": "BERT-NER-Simple"
            }
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))
    
    def simple_read(self, resume_text: str) -> Dict[str, Any]:
        """Lightweight read for upload"""
        if not resume_text:
            return {"error": "Empty resume text", "status": "FAILED"}
        
        try:
            import hashlib
            resume_id = hashlib.md5(resume_text.encode()).hexdigest()[:12]
            
            contact_info = self.extract_contact_info(resume_text)
            name = self.extract_name(resume_text)
            
            return {
                "resume_id": resume_id,
                "name": name,
                "email": contact_info.get("email", "unknown@email.com"),
                "phone": contact_info["phone"][0] if contact_info["phone"] else "",
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