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
        """Extract skills - COMPREHENSIVE patterns including BI, data analysis tools"""
        skills = set()
        
        # Comprehensive tech skills patterns
        skill_patterns = [
            # Programming Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|Scala|MATLAB|Perl|Groovy|Objective-C)\b',
            
            # Web Frameworks & Libraries
            r'\b(React|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Node\.js|Laravel|Rails|Next\.js|Svelte|Ember\.js|Backbone|Sinatra)\b',
            
            # ML/AI/NLP
            r'\b(PyTorch|TensorFlow|Keras|scikit-learn|XGBoost|BERT|GPT|Transformers|Hugging\s*Face|spaCy|NLTK|Scikit|OpenAI|LLM)\b',
            
            # Cloud Platforms
            r'\b(AWS|Azure|GCP|Google\s*Cloud|Heroku|DigitalOcean|IBM\s*Cloud|Oracle\s*Cloud|Alibaba\s*Cloud)\b',
            
            # AWS Services
            r'\b(EC2|S3|Lambda|RDS|DynamoDB|SageMaker|CloudFormation|IAM|VPC|ECS|EKS|SNS|SQS)\b',
            
            # DevOps & Deployment
            r'\b(Docker|Kubernetes|Jenkins|GitLab|GitHub\s*Actions|Terraform|Ansible|Prometheus|Grafana|ELK\s*Stack|Elasticsearch|Logstash|Kibana)\b',
            
            # Databases
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|DynamoDB|Snowflake|Oracle|MSSQL|MariaDB|Cassandra|Neo4j)\b',
            
            # Big Data & Data Processing
            r'\b(Spark|Hadoop|Kafka|Airflow|Pandas|NumPy|Dask|PySpark|Hive|Pig)\b',
            
            # BI & Data Visualization Tools
            r'\b(Power\s*BI|Tableau|Looker|Qlik|Microstrategy|SAP\s*Analytics|Informatica|Alteryx|Excel|Google\s*Data\s*Studio|matplotlib|plotly|seaborn|ggplot2)\b',
            
            # Office & Productivity
            r'\b(Excel|Word|PowerPoint|Outlook|Access|Project|Visio|SharePoint|OneNote|Google\s*Sheets|Google\s*Docs|Google\s*Slides)\b',
            
            # Frontend Technologies
            r'\b(HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|AJAX|Webpack|Gulp|Grunt|Parcel|Sass|PostCSS|BEM)\b',
            
            # Testing Frameworks
            r'\b(pytest|Jest|Selenium|Cypress|Mocha|Jasmine|RSpec|JUnit|TestNG|Cucumber|Mockito|Junit5)\b',
            
            # Version Control & Collaboration
            r'\b(Git|GitHub|GitLab|Bitbucket|SVN|Mercurial|Jira|Confluence|Trello|Asana|Monday\.com)\b',
            
            # API & Integration
            r'\b(REST|GraphQL|SOAP|Microservices|API|gRPC|WebSockets|Message\s*Queue|ActiveMQ|RabbitMQ|Kafka)\b',
            
            # Statistical & Analysis Tools
            r'\b(SPSS|SAS|R|Minitab|JMP|Stata|Jupyter|RStudio|Anaconda|Databricks|Colab)\b',
            
            # Mobile Development
            r'\b(React\s*Native|Flutter|Swift|Kotlin|Objective-C|Xamarin|Ionic|Cordova|Android|iOS)\b',
            
            # IDE & Development Tools
            r'\b(VS\s*Code|Visual\s*Studio|IntelliJ|PyCharm|Eclipse|Sublime|Atom|Vim|Emacs|Xcode)\b',
            
            # Agile & Methodologies
            r'\b(Agile|Scrum|Kanban|Waterfall|Lean|CI/CD|DevOps|TDD|BDD|OOP|SOLID|Design\s*Patterns)\b',
            
            # Cloud Storage & CDN
            r'\b(AWS\s*S3|Google\s*Cloud\s*Storage|Azure\s*Blob|Dropbox|OneDrive|Box|GCS|CloudFront|CloudFlare)\b',
            
            # Machine Learning Operations
            r'\b(MLOps|MLflow|Kubeflow|Airflow|DVC|Weights\s*&\s*Biases|Neptune|Comet)\b',
            
            # Container & Orchestration
            r'\b(Docker|Kubernetes|OpenShift|Nomad|Swarm|ECS|EKS|GKE|AKS)\b',
            
            # Database Management Tools
            r'\b(DBeaver|Navicat|Workbench|SQL\s*Server\s*Management|pgAdmin|MongoDB\s*Compass|Redis\s*Commander)\b',
            
            # Communication & Chat
            r'\b(Slack|Microsoft\s*Teams|Discord|Zoom|Skype|Slack\s*Bot|Teams\s*Bot)\b',
            
            # Other Important Tools
            r'\b(Postman|Insomnia|SoapUI|Git\s*Bash|PowerShell|Bash|Linux|Unix|Windows|macOS)\b'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([m.strip() for m in matches if m])
        
        return sorted(list(skills))
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education"""
        education = []
        
        degree_patterns = [
            (r'\b(PhD|Ph\.D|Doctor\s+of\s+Philosophy)\b', 'PhD'),
            (r'\b(Master|MS|MSc|M\.S|MA|MBA)\b', 'Master'),
            (r'\b(Bachelor|BS|BSc|B\.S|BA|BTech)\b', 'Bachelor'),
            (r'\b(Associate|AS)\b', 'Associate'),
        ]
        
        lines = text.split('\n')
        found_degrees = set()
        
        for i, line in enumerate(lines):
            for pattern, degree_type in degree_patterns:
                if re.search(pattern, line, re.IGNORECASE) and degree_type not in found_degrees:
                    # Extract field of study from the same line
                    field = "General"
                    field_match = re.search(r'(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', line)
                    if field_match:
                        field = field_match.group(1).strip()
                    
                    # Get year from nearby context
                    context = ' '.join(lines[max(0, i-1):min(len(lines), i+3)])
                    year_match = re.search(r'\b(19|20)\d{2}\b', context)
                    
                    education.append({
                        "degree": f"{degree_type} in {field}",
                        "institution": "Unknown",
                        "year": year_match.group(0) if year_match else "Unknown"
                    })
                    found_degrees.add(degree_type)
                    break
        
        return education
    
    def extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract experience years and companies"""
        # Years
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:?\s*(\d+)\+?\s*years?',
            r'\b(\d+)\+?\s*yrs?\b',
        ]
        
        years = 0
        for pattern in years_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    years = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
        
        # Companies - simple pattern matching
        companies = []
        company_patterns = [
            r'(?:at|worked at|worked for)\s+([A-Z][A-Za-z\s&]+?)(?:\.|,|\n)',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend([m.strip() for m in matches if m.strip()])
        
        companies = list(dict.fromkeys(companies))[:5]
        
        # Job titles - more flexible patterns
        title_patterns = [
            r'(?:Title|Position|Role)[:]*\s+([A-Za-z\s]+?)(?:,|\n|$)',
            r'\b(Senior|Lead|Principal|Manager)\s+(Software\s+)?(Engineer|Developer|Scientist)',
            r'\b(Data|Machine\s+Learning|Full\s+Stack)\s+(Engineer|Scientist|Developer)',
        ]
        
        roles = set()
        for pattern in title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    role = ' '.join([m.strip() for m in match if m]).strip()
                else:
                    role = match.strip() if isinstance(match, str) else ''
                
                if role and 5 < len(role) < 100:
                    roles.add(role)
        
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
                "education": [e["degree"] for e in education] if education else [],
                "total_experience_(months)": experience.get("years", 0) * 12,
                "current_company_name": experience.get("companies", [])[0] if experience.get("companies") else "Unknown",
                "current_location": "Unknown",
                "designation": experience.get("roles", []),
                "relevant_experience_(primary)": {"job_history": []},
                "relevant_experience_(secondary)": {"projects": [], "certifications": certs},
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
    
    # ============== NER VISUALIZATION METHODS ==============
    
    def get_entity_table(self, resume_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a formatted table of extracted entities
        
        Args:
            resume_data: Extracted resume data with entities
            
        Returns:
            List of dicts with Category, Value, Confidence
        """
        if not isinstance(resume_data, dict):
            return []
        
        entity_table = []
        
        # Skills
        primary_skills = resume_data.get("primary_skills", [])
        if isinstance(primary_skills, list):
            for skill in primary_skills:
                entity_table.append({
                    "Category": "🛠️ Skill (Primary)",
                    "Value": str(skill),
                    "Confidence": "High"
                })
        
        secondary_skills = resume_data.get("secondary_skills", [])
        if isinstance(secondary_skills, list):
            for skill in secondary_skills:
                entity_table.append({
                    "Category": "🛠️ Skill (Secondary)",
                    "Value": str(skill),
                    "Confidence": "Medium"
                })
        
        # Education
        education = resume_data.get("education", [])
        if isinstance(education, list):
            for edu in education:
                edu_str = edu.get("degree", str(edu)) if isinstance(edu, dict) else str(edu)
                entity_table.append({
                    "Category": "🎓 Education",
                    "Value": edu_str,
                    "Confidence": "High"
                })
        
        # Experience/Designations
        designation = resume_data.get("designation", [])
        if isinstance(designation, list):
            for job_title in designation:
                entity_table.append({
                    "Category": "💼 Job Title",
                    "Value": str(job_title),
                    "Confidence": "High"
                })
        
        # Certifications
        sec_exp = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(sec_exp, dict):
            certifications = sec_exp.get("certifications", [])
            if isinstance(certifications, list):
                for cert in certifications:
                    entity_table.append({
                        "Category": "📜 Certification",
                        "Value": str(cert),
                        "Confidence": "Medium"
                    })
        
        # Company
        company = resume_data.get("current_company_name", "")
        if company and company != "Unknown":
            entity_table.append({
                "Category": "🏢 Company",
                "Value": str(company),
                "Confidence": "High"
            })
        
        # Contact Info
        email = resume_data.get("email_address", "")
        if email and email != "unknown@email.com":
            entity_table.append({
                "Category": "📧 Email",
                "Value": str(email),
                "Confidence": "High"
            })
        
        contact_number = resume_data.get("contact_number", [])
        if isinstance(contact_number, list):
            for phone in contact_number:
                entity_table.append({
                    "Category": "📱 Phone",
                    "Value": str(phone),
                    "Confidence": "High"
                })
        
        return entity_table
    
    def highlight_entities_in_text(self, resume_text: str, resume_data: Dict[str, Any]) -> str:
        """
        Create HTML with highlighted entities in the resume text
        
        Args:
            resume_text: Original resume text
            resume_data: Extracted resume data
            
        Returns:
            HTML string with highlighted entities
        """
        if not isinstance(resume_data, dict) or not resume_text:
            return resume_text
        
        # Define colors for different entity types
        colors = {
            "Skill": "#FFD700",           # Gold
            "Education": "#87CEEB",        # Sky Blue
            "Job Title": "#90EE90",        # Light Green
            "Company": "#DDA0DD",          # Plum
            "Email": "#FFB6C1",            # Light Pink
            "Phone": "#FFA07A",            # Light Salmon
            "Certification": "#F0E68C"     # Khaki
        }
        
        highlighted_text = resume_text
        entities_to_highlight = []
        
        # Collect all entities with their types
        for skill in resume_data.get("primary_skills", []):
            entities_to_highlight.append((str(skill), "Skill"))
        
        for edu in resume_data.get("education", []):
            edu_str = edu.get("degree", str(edu)) if isinstance(edu, dict) else str(edu)
            entities_to_highlight.append((edu_str, "Education"))
        
        for job_title in resume_data.get("designation", []):
            entities_to_highlight.append((str(job_title), "Job Title"))
        
        if resume_data.get("current_company_name") and resume_data.get("current_company_name") != "Unknown":
            entities_to_highlight.append((resume_data.get("current_company_name"), "Company"))
        
        sec_exp = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(sec_exp, dict):
            for cert in sec_exp.get("certifications", []):
                entities_to_highlight.append((str(cert), "Certification"))
        
        # Sort by length (longest first) to avoid partial highlighting
        entities_to_highlight = sorted(set(entities_to_highlight), key=lambda x: len(x[0]), reverse=True)
        
        # Highlight each entity (case-insensitive)
        highlighted_positions = set()
        for entity_text, entity_type in entities_to_highlight:
            color = colors.get(entity_type, "#FFFF00")
            
            # Case-insensitive search
            pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
            
            def replace_func(match):
                start = match.start()
                end = match.end()
                
                # Check if this position was already highlighted
                if any(start <= pos < end or pos in range(start, end) for pos in highlighted_positions):
                    return match.group(0)
                
                for pos in range(start, end):
                    highlighted_positions.add(pos)
                
                return f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{match.group(0)}</mark>'
            
            highlighted_text = pattern.sub(replace_func, highlighted_text)
        
        return highlighted_text
    
    def get_entity_summary(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of extracted entities with counts
        
        Args:
            resume_data: Extracted resume data
            
        Returns:
            Dictionary with entity counts and statistics
        """
        if not isinstance(resume_data, dict):
            return {}
        
        primary_skills = resume_data.get("primary_skills", [])
        secondary_skills = resume_data.get("secondary_skills", [])
        education = resume_data.get("education", [])
        designation = resume_data.get("designation", [])
        
        sec_exp = resume_data.get("relevant_experience_(secondary)", {})
        certifications = sec_exp.get("certifications", []) if isinstance(sec_exp, dict) else []
        
        contact_number = resume_data.get("contact_number", [])
        
        summary = {
            "total_entities_extracted": 0,
            "skills_count": {
                "primary": len(primary_skills) if isinstance(primary_skills, list) else 0,
                "secondary": len(secondary_skills) if isinstance(secondary_skills, list) else 0
            },
            "education_count": len(education) if isinstance(education, list) else 0,
            "job_titles_count": len(designation) if isinstance(designation, list) else 0,
            "certifications_count": len(certifications) if isinstance(certifications, list) else 0,
            "contact_info_count": len(contact_number) if isinstance(contact_number, list) else 0,
            "has_email": bool(resume_data.get("email_address") and resume_data.get("email_address") != "unknown@email.com"),
            "extraction_quality": "COMPLETE" if resume_data.get("extraction_status") == "SUCCESS" else "PARTIAL"
        }
        
        # Calculate total
        summary["total_entities_extracted"] = (
            summary["skills_count"]["primary"] +
            summary["skills_count"]["secondary"] +
            summary["education_count"] +
            summary["job_titles_count"] +
            summary["certifications_count"] +
            summary["contact_info_count"]
        )
        
        return summary
    
    def get_visualization_data(self, resume_data: Dict[str, Any], resume_text: str = "") -> Dict[str, Any]:
        """
        Get complete visualization data for frontend
        
        Args:
            resume_data: Extracted resume data
            resume_text: Original resume text
            
        Returns:
            Dictionary with all visualization components
        """
        if not isinstance(resume_data, dict):
            resume_data = {}
        
        return {
            "entity_table": self.get_entity_table(resume_data),
            "entity_summary": self.get_entity_summary(resume_data),
            "highlighted_text": self.highlight_entities_in_text(resume_text, resume_data) if resume_text else "",
            "extraction_status": resume_data.get("extraction_status", "UNKNOWN"),
            "model_type": "BERT-NER-Visualization"
        }