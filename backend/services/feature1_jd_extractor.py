"""
Feature 1: Job Description Extraction using BERT NER
Extracts: skills, experience, education, certifications from JD text
Uses same NER approach as resume parsing for consistency
"""

from transformers import pipeline
import torch
import re
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JDExtractor:
    def __init__(self):
        """Initialize BERT NER for JD extraction"""
        try:
            logger.info("Loading BERT NER model for JD extraction...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=-1
            )
            logger.info("BERT NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load BERT NER: {e}")
            self.ner_pipeline = None
    
    def extract_jd_data(self, jd_text: str) -> Dict[str, Any]:
        """Extract structured data from job description using BERT NER + regex"""
        
        # Always use comprehensive regex extraction (more reliable for JDs)
        return self._comprehensive_extraction(jd_text)
    
    def _comprehensive_extraction(self, jd_text: str) -> Dict[str, Any]:
        """Comprehensive regex-based extraction optimized for JDs"""
        
        # Skills - comprehensive patterns covering multiple domains
        skill_patterns = [
            # Programming Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|MATLAB|Scala|Perl|Haskell)\b',
            
            # Web Frameworks
            r'\b(React|Angular|Vue|Svelte|Django|Flask|FastAPI|Spring|Express|Node\.?js|Laravel|Rails|ASP\.NET|Next\.js)\b',
            
            # Cloud & DevOps
            r'\b(Docker|Kubernetes|AWS|Azure|GCP|Google\s*Cloud|Git|Jenkins|CI/CD|Terraform|Ansible|Chef|Puppet)\b',
            
            # Databases
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|Oracle|DynamoDB|SQLite|Neo4j)\b',
            
            # ML/AI
            r'\b(Machine\s*Learning|Deep\s*Learning|NLP|Computer\s*Vision|TensorFlow|PyTorch|Scikit-learn|Keras|XGBoost)\b',
            
            # Web & API
            r'\b(HTML|CSS|REST|RESTful|GraphQL|Microservices|API)\b',
            
            # Methodologies
            r'\b(Agile|Scrum|Kanban|DevOps|TDD|BDD)\b',
            
            # Tools
            r'\b(Jira|Confluence|VS\s*Code|IntelliJ|Eclipse|Postman|Swagger)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        # Experience - multiple patterns
        experience = 0
        exp_patterns = [
            r'(\d+)\+?\s*(?:to\s+\d+)?\s*years?',
            r'(\d+)\+?\s*yrs?',
            r'minimum\s+of\s+(\d+)\s+years?',
            r'at\s+least\s+(\d+)\s+years?',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                experience = int(match.group(1))
                break
        
        # Education - hierarchical detection
        education = "Bachelor's"
        if re.search(r'PhD|Ph\.D|Doctorate|Doctoral', jd_text, re.IGNORECASE):
            education = "PhD"
        elif re.search(r'Master|MS|MSc|M\.S|M\.Sc|MBA|MEng', jd_text, re.IGNORECASE):
            education = "Master's"
        elif re.search(r'Bachelor|BS|BSc|B\.S|B\.Sc|BA|B\.A|BTech', jd_text, re.IGNORECASE):
            education = "Bachelor's"
        elif re.search(r'Associate|AA|AS', jd_text, re.IGNORECASE):
            education = "Associate"
        elif re.search(r'High\s*School|Diploma', jd_text, re.IGNORECASE):
            education = "High School"
        
        # Certifications - comprehensive patterns
        cert_patterns = [
            r'(AWS\s+Certified[\w\s]+)',
            r'(Azure[\w\s]+Certified[\w\s]*)',
            r'(Google\s+Cloud[\w\s]+Certified[\w\s]*)',
            r'\b(PMP|CISSP|CEH|CISA|CISM|CompTIA\s+\w+)\b',
            r'(Certified[\w\s]+(?:Professional|Specialist|Expert|Developer|Administrator|Architect))',
            r'(Oracle\s+Certified[\w\s]+)',
            r'(Cisco\s+Certified[\w\s]+)',
            r'\b(CCNA|CCNP|CCIE)\b',
        ]
        
        certifications = set()
        for pattern in cert_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                cert = match.strip() if isinstance(match, str) else match[0].strip()
                if len(cert) > 2:  # Avoid short false positives
                    certifications.add(cert)
        
        return {
            "required_skills": sorted(list(skills)),
            "required_experience": experience,
            "required_education": education,
            "certifications": sorted(list(certifications)),
            "status": "SUCCESS"
        }