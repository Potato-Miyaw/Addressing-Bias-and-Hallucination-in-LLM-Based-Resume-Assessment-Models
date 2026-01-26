"""
Feature 2: BERT NER for Resume Parsing
Fine-tuned BERT model for token-level classification with BIO tagging
Extracts: SKILLS, EDUCATION, EXPERIENCE, CERTIFICATIONS
Based on Resume_BERT_NER_Modeling_Task documentation
"""

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    pipeline
)
from typing import List, Dict, Any
import torch
import re
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeNERExtractor:
    def __init__(self, model_path: str = None):
        """
        Initialize BERT NER for resume parsing
        Uses fine-tuned BertForTokenClassification if available,
        otherwise falls back to regex + generic NER
        """
        # Resume-specific entity schema (BIO tagging)
        self.label_list = [
            "O",
            "B-NAME", "I-NAME",
            "B-DESIGNATION", "I-DESIGNATION",
            "B-PHONE", "I-PHONE",
            "B-EMAIL", "I-EMAIL",
            "B-EDU", "I-EDU",
            "B-COMPANY", "I-COMPANY",
            "B-LOCATION", "I-LOCATION",
            "B-SKILL-PRIMARY", "I-SKILL-PRIMARY",
            "B-SKILL-SECONDARY", "I-SKILL-SECONDARY",
            "B-EXP-MONTHS", "I-EXP-MONTHS",
            "B-JOB-TITLE", "I-JOB-TITLE",
            "B-PROJECT", "I-PROJECT",
            "B-CERT", "I-CERT",
        ]
        
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        
        # Try to load fine-tuned model
        self.model = None
        self.tokenizer = None
        self.ner_pipeline = None
        
        if model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading fine-tuned resume NER model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path)
                self.ner_pipeline = pipeline(
                    "ner",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=-1
                )
                logger.info("Fine-tuned resume NER model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}")
                self.model = None
        
        # Fallback: Try generic NER model
        if self.ner_pipeline is None:
            try:
                logger.info("Loading fallback generic NER model: dslim/bert-base-NER")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=-1
                )
                logger.info("Generic NER model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                self.ner_pipeline = None
    
    def extract_entities_bert(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using BERT NER with BIO tagging
        Returns entities with types: SKILL, EDU, EXP, CERT
        """
        if not self.ner_pipeline or not text:
            return []
        
        try:
            entities = self.ner_pipeline(text)
            
            formatted_entities = []
            for entity in entities:
                # Map entity types
                entity_type = entity.get("entity_group", entity.get("entity", "O"))
                
                # Clean BIO prefix if present (B-SKILL -> SKILL)
                if entity_type.startswith("B-") or entity_type.startswith("I-"):
                    entity_type = entity_type[2:]
                
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
    
    def merge_tokens(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Merge BIO-tagged tokens into complete entities
        Converts token-level predictions to entity-level extractions
        """
        merged = {
            "NAME": [],
            "DESIGNATION": [],
            "PHONE": [],
            "EMAIL": [],
            "EDU": [],
            "COMPANY": [],
            "LOCATION": [],
            "SKILL-PRIMARY": [],
            "SKILL-SECONDARY": [],
            "EXP-MONTHS": [],
            "JOB-TITLE": [],
            "PROJECT": [],
            "CERT": []
        }
        
        current_entity = None
        current_type = None
        
        for entity in entities:
            entity_type = entity["entity_type"]
            value = entity["value"]
            
            if entity_type in merged:
                # Start new entity or continue current
                if current_type == entity_type:
                    current_entity += " " + value
                else:
                    if current_entity and current_type:
                        merged[current_type].append(current_entity.strip())
                    current_entity = value
                    current_type = entity_type
        
        # Add last entity
        if current_entity and current_type:
            merged[current_type].append(current_entity.strip())
        
        return merged
    
    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract contact information: phone numbers and email"""
        contact = {
            "phone": [],
            "email": None
        }
        
        # Phone patterns
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # +1 (555) 123-4567
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 555-123-4567
            r'\d{10}',  # 5551234567
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            contact["phone"].extend([m.strip() for m in matches])
        
        # Remove duplicates
        contact["phone"] = list(dict.fromkeys(contact["phone"]))[:3]
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text)
        if email_match:
            contact["email"] = email_match.group(0)
        
        return contact
    
    def extract_location(self, text: str) -> str:
        """Extract current location"""
        # Pattern for city, state/country
        location_patterns = [
            r'(?:based|located|living|residing)\s+(?:in|at)\s+([A-Z][a-zA-Z\s]+(?:,\s*[A-Z]{2})?)',
            r'(?:Current\s+)?[Ll]ocation\s*:\s*([A-Z][a-zA-Z\s,]+)',
            r'\b([A-Z][a-zA-Z]+,\s*[A-Z]{2})\b',  # City, ST format
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback to BERT entities
        entities = self.extract_entities_bert(text)
        for e in entities:
            if e["entity_type"] in ["LOCATION", "LOC"] and e["score"] > 0.7:
                return e["value"]
        
        return "Unknown"
    
    def extract_job_history(self, text: str) -> List[Dict[str, str]]:
        """Extract job history with titles, companies, and descriptions"""
        job_history = []
        
        # Extract job titles
        title_patterns = [
            r'\b(Senior|Lead|Principal|Staff|Chief)\s+(Engineer|Developer|Scientist|Architect|Manager|Analyst)',
            r'\b(Software|Machine\s+Learning|Data|Backend|Frontend|Full\s*Stack|DevOps)\s+(Engineer|Developer)',
        ]
        
        # Get companies from BERT
        entities = self.extract_entities_bert(text)
        companies = [e["value"] for e in entities if e["entity_type"] in ["COMPANY", "ORGANIZATION", "ORG"] and e["score"] > 0.6]
        
        # Extract roles
        roles = set()
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    roles.add(' '.join(match))
                else:
                    roles.add(match)
        
        # Pair roles with companies (simplified - better with structured parsing)
        for i, role in enumerate(list(roles)[:3]):
            company = companies[i] if i < len(companies) else "Unknown"
            job_history.append({
                "job_title": role,
                "job_company": company,
                "job_description": ""  # Requires more context to extract
            })
        
        return job_history
    
    def extract_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract project information"""
        projects = []
        
        # Project keywords
        project_patterns = [
            r'[Pp]roject[s]?\s*:\s*([A-Z][^\n.;]+)',
            r'[Dd]eveloped\s+([A-Z][a-zA-Z\s]+(?:Platform|System|Application|Tool|Dashboard))',
            r'[Bb]uilt\s+([A-Z][a-zA-Z\s]+(?:Platform|System|Application|Tool|Dashboard))',
        ]
        
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                project_name = match.strip() if isinstance(match, str) else match[0].strip()
                if len(project_name) > 5:
                    projects.append({
                        "project_name": project_name,
                        "project_company": "Unknown",
                        "project_description": ""
                    })
        
        return projects[:5]
    
    def extract_certifications_detailed(self, text: str) -> List[Dict[str, str]]:
        """Extract certifications with providers"""
        certifications = []
        
        cert_patterns = [
            (r'(AWS\s+Certified[^.\n]+)', 'AWS'),
            (r'(Azure[^.\n]+Certified[^.\n]+)', 'Microsoft'),
            (r'(Google\s+Cloud[^.\n]+Certified[^.\n]+)', 'Google'),
            (r'\b(PMP)', 'PMI'),
            (r'\b(CISSP|CEH|CISA)', 'Various'),
            (r'(Certified[^.\n]+(?:Professional|Specialist|Expert|Administrator|Developer))', 'Various'),
        ]
        
        for pattern, provider in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cert_title = match.strip() if isinstance(match, str) else match[0].strip()
                if len(cert_title) > 3:
                    certifications.append({
                        "certificate_title": cert_title,
                        "certification_provider": provider
                    })
        
        return certifications
    
    def extract_total_experience_months(self, text: str) -> int:
        """Extract total experience in months"""
        # Extract years of experience
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+(?:in|as|working)',
        ]
        
        years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                break
        
        return years * 12  # Convert to months
    
    def extract_applicant_description(self, text: str) -> str:
        """Generate a summary description of the applicant"""
        # Extract first few sentences or summary section
        summary_patterns = [
            r'(?:Summary|Profile|About|Overview)\s*:?\s*([^\n]+(?:\n[^\n]+){0,3})',
            r'^([A-Z][^\n]+(?:\n[A-Z][^\n]+){0,2})',
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                description = match.group(1).strip()
                if len(description) > 50:
                    return description[:500]  # Limit to 500 chars
        
        # Fallback: Use first 200 chars
        return text[:200].strip() + "..."
    
    def extract_name(self, text: str) -> str:
        """
        Extract candidate name from resume
        Strategy: First PERSON entity or first line with name pattern
        """
        # Try BERT NER first
        entities = self.extract_entities_bert(text)
        for entity in entities:
            if entity["entity_type"] == "PERSON" and entity["score"] > 0.7:
                # Clean up subword tokens
                name = entity["value"].replace(" ##", "").replace("##", "")
                if len(name.split()) >= 2:  # At least first and last name
                    return name
        
        # Fallback: Look for name patterns in first 3 lines
        lines = text.split('\n')[:3]
        for line in lines:
            line = line.strip()
            # Name pattern: 2-4 capitalized words, no symbols
            name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$'
            match = re.match(name_pattern, line)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract technical skills using comprehensive patterns + BERT
        """
        skills = set()
        
        # Comprehensive skill patterns
        skill_patterns = [
            # Programming Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|MATLAB|Scala|Perl|Haskell|Clojure|Elixir)\b',
            
            # Web Frameworks
            r'\b(React|Angular|Vue|Svelte|Django|Flask|FastAPI|Spring|Express|Node\.js|Laravel|Rails|ASP\.NET|Next\.js|Nuxt\.js)\b',
            
            # ML/AI Frameworks
            r'\b(PyTorch|TensorFlow|JAX|Keras|scikit-learn|XGBoost|LightGBM|Hugging\s*Face|Transformers|BERT|GPT|spaCy|NLTK)\b',
            
            # Cloud Platforms
            r'\b(AWS|Azure|GCP|Google\s*Cloud|Amazon\s*Web\s*Services|Microsoft\s*Azure|Heroku|DigitalOcean)\b',
            
            # DevOps & Tools
            r'\b(Docker|Kubernetes|Jenkins|GitLab|GitHub\s*Actions|CircleCI|Travis\s*CI|Terraform|Ansible|Chef|Puppet)\b',
            
            # Databases
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|DynamoDB|Oracle|SQLite|Neo4j|Snowflake)\b',
            
            # Data & Big Data
            r'\b(Spark|Hadoop|Kafka|Airflow|Pandas|NumPy|Dask|Ray|Flink|Storm)\b',
            
            # Frontend
            r'\b(HTML|CSS|SASS|SCSS|Bootstrap|Tailwind|Material\s*UI|jQuery|Webpack|Babel)\b',
            
            # Testing
            r'\b(pytest|Jest|Mocha|Selenium|Cypress|JUnit|TestNG)\b',
            
            # Other Tools
            r'\b(Git|Jira|Confluence|VS\s*Code|IntelliJ|Eclipse|Postman|Swagger)\b',
            
            # Methodologies
            r'\b(Agile|Scrum|Kanban|CI/CD|DevOps|Microservices|REST|GraphQL|gRPC)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        # Also check BERT entities
        entities = self.extract_entities_bert(text)
        for entity in entities:
            if entity["entity_type"] in ["ORGANIZATION", "MISC"] and entity["score"] > 0.8:
                value = entity["value"].replace(" ##", "").replace("##", "")
                # Check if it's a known tech term
                if any(tech.lower() in value.lower() for tech in ["docker", "aws", "python", "react", "kubernetes"]):
                    skills.add(value)
        
        return sorted(list(skills))
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        Extract education using patterns + BERT entities
        """
        education = []
        
        # Degree patterns
        degree_patterns = [
            r'\b(PhD|Ph\.D|Doctor\s+of\s+Philosophy)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
            r'\b(Master|MS|MSc|M\.S|M\.Sc|MA|M\.A|MBA|MEng)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
            r'\b(Bachelor|BS|BSc|B\.S|B\.Sc|BA|B\.A|BTech|B\.Tech)\b.*?(?:in\s+)?([A-Z][a-zA-Z\s]+)',
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for pattern in degree_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    degree = match.group(1)
                    field = match.group(2).strip() if len(match.groups()) > 1 else "Unknown"
                    
                    # Extract institution (check nearby lines)
                    context = ' '.join(lines[max(0, i-1):min(len(lines), i+3)])
                    context_entities = self.extract_entities_bert(context)
                    
                    institutions = [
                        e["value"].replace(" ##", "").replace("##", "")
                        for e in context_entities
                        if e["entity_type"] == "ORGANIZATION" and e["score"] > 0.5
                    ]
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', context)
                    
                    education.append({
                        "degree": f"{degree} in {field}",
                        "institution": institutions[0] if institutions else "Unknown",
                        "year": year_match.group(0) if year_match else "Unknown"
                    })
                    break
        
        return education
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """
        Extract work experience details
        """
        # Extract years of experience
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+(?:in|as|working)'
        ]
        
        years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                years = int(match.group(1))
                break
        
        # Extract companies using BERT
        entities = self.extract_entities_bert(text)
        companies = []
        for e in entities:
            if e["entity_type"] == "ORGANIZATION" and e["score"] > 0.6:
                company = e["value"].replace(" ##", "").replace("##", "")
                # Filter out obvious non-companies
                if company not in ["University", "College", "Institute"] and len(company) > 2:
                    companies.append(company)
        
        # Remove duplicates, keep first 5
        companies = list(dict.fromkeys(companies))[:5]
        
        # Extract job titles
        title_patterns = [
            r'\b(Senior|Lead|Principal|Staff|Chief)\s+(Engineer|Developer|Scientist|Architect|Manager|Analyst)',
            r'\b(Engineer|Developer|Scientist|Architect|Manager|Analyst|Consultant|Director|Specialist)\b',
            r'\b(Software|Machine\s+Learning|Data|Backend|Frontend|Full\s*Stack|DevOps)\s+(Engineer|Developer)'
        ]
        
        roles = set()
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    roles.add(' '.join(match))
                else:
                    roles.add(match)
        
        return {
            "years": years,
            "companies": companies,
            "roles": list(roles)[:5]
        }
    
    def extract_certifications(self, text: str) -> List[str]:
        """
        Extract professional certifications
        """
        certifications = []
        
        cert_patterns = [
            r'(AWS\s+Certified[^.\n]+)',
            r'(Azure[^.\n]+Certified[^.\n]+)',
            r'(Google\s+Cloud[^.\n]+Certified[^.\n]+)',
            r'\b(PMP|CISSP|CEH|CISA|CompTIA\s+\w+)',
            r'(Certified[^.\n]+(?:Professional|Specialist|Expert|Administrator|Developer))',
            r'(\w+\s+Certified\s+\w+)',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cert = match.strip() if isinstance(match, str) else match[0].strip()
                if len(cert) > 5:  # Avoid short false positives
                    certifications.append(cert)
        
        # Remove duplicates
        return list(dict.fromkeys(certifications))
    
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Complete resume parsing pipeline using BERT NER
        
        Architecture:
        Resume Text → BERT Tokenizer → BERT Encoder → Token Classification → BIO Labels → Structured JSON
        
        Returns comprehensive structured extraction matching the required schema
        """
        if not resume_text:
            return self._empty_result("Empty resume text")
        
        try:
            # Step 1: Extract raw BERT entities
            raw_entities = self.extract_entities_bert(resume_text)
            
            # Step 2: Merge BIO-tagged tokens
            merged_entities = self.merge_tokens(raw_entities)
            
            # Step 3: Extract all fields (hybrid BERT + regex approach)
            
            # Basic info
            name_list = merged_entities.get("NAME", [])
            name = name_list[0] if name_list else self.extract_name(resume_text)
            
            designation = merged_entities.get("DESIGNATION", []) or merged_entities.get("JOB-TITLE", [])
            
            # Contact info
            contact_info = self.extract_contact_info(resume_text)
            contact_number = merged_entities.get("PHONE", []) or contact_info["phone"]
            
            email_list = merged_entities.get("EMAIL", [])
            email_address = email_list[0] if email_list else (contact_info.get("email") or "unknown@email.com")
            
            # Education
            edu_bert = merged_entities.get("EDU", [])
            edu_regex = self.extract_education(resume_text)
            education = [e["degree"] for e in edu_regex] if edu_regex else edu_bert
            
            # Company and location
            company_list = merged_entities.get("COMPANY", [])
            current_company_name = company_list[0] if company_list else "Unknown"
            
            location_list = merged_entities.get("LOCATION", [])
            current_location = location_list[0] if location_list else self.extract_location(resume_text)
            
            # Skills
            primary_skills_bert = merged_entities.get("SKILL-PRIMARY", [])
            secondary_skills_bert = merged_entities.get("SKILL-SECONDARY", [])
            all_skills_regex = self.extract_skills(resume_text)
            
            # Combine and categorize skills
            if not primary_skills_bert and not secondary_skills_bert:
                # No BERT skills, split regex skills
                primary_skills = all_skills_regex[:10]  # Top 10 as primary
                secondary_skills = all_skills_regex[10:20]  # Next 10 as secondary
            else:
                primary_skills = list(set(primary_skills_bert + all_skills_regex[:5]))
                secondary_skills = list(set(secondary_skills_bert))
            
            # Experience
            total_experience_months = self.extract_total_experience_months(resume_text)
            
            # Job history
            job_history = self.extract_job_history(resume_text)
            
            # Projects and certifications
            projects = self.extract_projects(resume_text)
            certifications = self.extract_certifications_detailed(resume_text)
            
            # Applicant description
            applicant_description = self.extract_applicant_description(resume_text)
            
            return {
                "name": name,
                "designation": designation[:3] if designation else [],  # Limit to top 3
                "contact_number": contact_number[:3] if contact_number else [],
                "email_address": email_address,
                "education": education[:5] if education else [],
                "current_company_name": current_company_name,
                "current_location": current_location,
                "primary_skills": primary_skills[:15] if primary_skills else [],
                "secondary_skills": secondary_skills[:15] if secondary_skills else [],
                "total_experience_(months)": total_experience_months,
                "relevant_experience_(primary)": {
                    "job_history": job_history
                },
                "relevant_experience_(secondary)": {
                    "projects": projects,
                    "certifications": certifications
                },
                "applicant_description": applicant_description,
                "entities": raw_entities,
                "extraction_status": "SUCCESS",
                "model_type": "BERT-NER-BIO" if self.model else "BERT-NER-Generic+Regex"
            }
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))
    
    def simple_read(self, resume_text: str) -> Dict[str, Any]:
        """
        Lightweight resume reading - extract basic info without full NER
        Use this for initial upload/storage, run full NER during matching
        """
        if not resume_text:
            return {"error": "Empty resume text", "status": "FAILED"}
        
        try:
            import hashlib
            resume_id = hashlib.md5(resume_text.encode()).hexdigest()[:12]
            
            # Extract contact info
            contact_info = self.extract_contact_info(resume_text)
            
            # Extract name - MORE FLEXIBLE pattern
            lines = resume_text.split('\n')[:5]  # Check first 5 lines
            name = "Unknown"
            
            for line in lines:
                line = line.strip()
                if not line or len(line) < 4:
                    continue
                
                # More flexible name patterns
                name_patterns = [
                    r'^([A-Z][A-Z\s]+)$',  # All caps: SARAH CHEN
                    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',  # Title case: Sarah Chen
                    r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+[A-Z][a-z]+)$',  # With middle initial: Sarah M. Chen
                ]
                
                for pattern in name_patterns:
                    match = re.match(pattern, line)
                    if match:
                        name = match.group(1).strip()
                        # Validate it looks like a name (2-4 words, each 2+ chars)
                        words = name.split()
                        if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                            break
                
                if name != "Unknown":
                    break
            
            return {
                "resume_id": resume_id,
                "name": name,
                "email": contact_info.get("email", "unknown@email.com"),
                "phone": contact_info["phone"][0] if contact_info["phone"] else "",
                "text": resume_text,  # ✅ CRITICAL: Include full text
                "status": "STORED",
                "message": "Resume stored successfully. Full NER extraction during matching."
            }
        except Exception as e:
            logger.error(f"Simple read failed: {e}")
            return {"error": str(e), "status": "FAILED"}
    
    def _empty_result(self, error_msg: str = "Unknown error") -> Dict[str, Any]:
        """Return empty result structure on error"""
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
            "relevant_experience_(primary)": {
                "job_history": []
            },
            "relevant_experience_(secondary)": {
                "projects": [],
                "certifications": []
            },
            "applicant_description": "",
            "entities": [],
            "extraction_status": "FAILED",
            "error": error_msg
        }