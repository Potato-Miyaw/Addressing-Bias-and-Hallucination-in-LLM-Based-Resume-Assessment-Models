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
        
        # Skills - COMPREHENSIVE patterns covering ALL domains
        skill_patterns = [
            # Programming Languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|MATLAB|Scala|Perl|Haskell|Groovy|Dart)\b',
            
            # Web Frameworks & Libraries
            r'\b(React|Angular|Vue|Svelte|Django|Flask|FastAPI|Spring|Express|Node\.?js|Laravel|Rails|ASP\.NET|Next\.js|Ember|Backbone)\b',
            
            # Cloud & AWS Services
            r'\b(AWS|Azure|GCP|Google\s*Cloud|EC2|S3|Lambda|RDS|DynamoDB|SageMaker|CloudFormation|IAM|VPC|ECS|EKS)\b',
            
            # DevOps & Deployment
            r'\b(Docker|Kubernetes|Jenkins|GitLab|GitHub\s*Actions|Terraform|Ansible|Chef|Puppet|Prometheus|Grafana|ELK\s*Stack)\b',
            
            # Databases
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|Oracle|DynamoDB|SQLite|Neo4j|MariaDB|Snowflake|Hive)\b',
            
            # ML/AI/NLP
            r'\b(Machine\s*Learning|Deep\s*Learning|NLP|Computer\s*Vision|TensorFlow|PyTorch|Scikit-learn|Keras|XGBoost|BERT|GPT|Transformers|Hugging\s*Face|LightGBM)\b',
            
            # BI & Data Visualization (CRITICAL - ADDED)
            r'\b(Power\s*BI|Tableau|Looker|Qlik|Microstrategy|SAP\s*Analytics|Google\s*Data\s*Studio|Excel|matplotlib|plotly|seaborn|ggplot2)\b',
            
            # Data & Analytics Tools
            r'\b(Spark|Hadoop|Kafka|Airflow|Pandas|NumPy|SPSS|SAS|Stata|Alteryx|Informatica|Dask|PySpark)\b',
            
            # Web & API
            r'\b(HTML|CSS|REST|RESTful|GraphQL|Microservices|API|gRPC|SOAP|WebSocket)\b',
            
            # Testing Frameworks
            r'\b(pytest|Jest|Selenium|Cypress|Mocha|Jasmine|JUnit|TestNG|Cucumber|Mockito|RSpec)\b',
            
            # Version Control & Collaboration
            r'\b(Git|GitHub|GitLab|Bitbucket|SVN|Jira|Confluence|Trello|Asana|Monday|Slack)\b',
            
            # Office & Productivity
            r'\b(Excel|Word|PowerPoint|Outlook|Access|Google\s*Sheets|Google\s*Docs|Google\s*Slides|SharePoint|OneNote)\b',
            
            # IDEs & Development Tools
            r'\b(VS\s*Code|Visual\s*Studio|IntelliJ|PyCharm|Eclipse|Sublime|Jupyter|RStudio|Anaconda|Postman)\b',
            
            # Methodologies & Practices
            r'\b(Agile|Scrum|Kanban|Waterfall|Lean|CI/CD|DevOps|TDD|BDD|Microservices|OOP|SOLID)\b',
            
            # Mobile Development
            r'\b(React\s*Native|Flutter|Swift|Kotlin|Objective-C|Xamarin|Ionic|Android|iOS)\b',
            
            # Statistical & ML Ops Tools
            r'\b(MLOps|MLflow|Kubeflow|DVC|Weights\s*&\s*Biases|Neptune|Comet|Databricks)\b',
            
            # Cloud Storage & CDN
            r'\b(Google\s*Cloud\s*Storage|Azure\s*Blob|Dropbox|OneDrive|Box|CloudFront|CloudFlare)\b',
            
            # Communication Tools
            r'\b(Slack|Teams|Discord|Zoom|Skype)\b',
            
            # Database Management Tools
            r'\b(DBeaver|Navicat|Workbench|pgAdmin|MongoDB\s*Compass|Redis\s*Commander)\b',
            
            # Other Important Tools
            r'\b(Postman|Insomnia|SoapUI|Git\s*Bash|PowerShell|Bash|Linux|Unix|Windows)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            skills.update([m.strip() for m in matches if m])
        
        # Deduplicate case-insensitively (keep version with more capitals)
        skills_dict = {}
        for skill in skills:
            key = skill.lower()
            if key not in skills_dict or sum(1 for c in skill if c.isupper()) > sum(1 for c in skills_dict[key] if c.isupper()):
                skills_dict[key] = skill
        
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
            "required_skills": sorted(list(skills_dict.values())),
            "required_experience": experience,
            "required_education": education,
            "certifications": sorted(list(certifications)),
            "status": "SUCCESS"
        }