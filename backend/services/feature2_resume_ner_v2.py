"""
Feature 2B: Resume-Specific BERT NER
Uses yashpwr/resume-ner-bert-v2 - pre-trained on resume data
Alternative to generic dslim/bert-base-NER for comparison
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import re
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeNERExtractorV2:
    def __init__(self):
        """Initialize Resume-Specific BERT NER"""
        self.model_name = "yashpwr/resume-ner-bert-v2"
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        
        try:
            logger.info(f"Loading resume-specific NER model: {self.model_name}")
            
            # Try pipeline first (easier)
            self.ner_pipeline = pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
                device=-1
            )
            
            # Also load raw model for advanced usage
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            logger.info("✅ Resume-specific NER model loaded successfully")
            logger.info(f"Available labels: {list(self.model.config.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load resume-specific NER model: {e}")
            self.ner_pipeline = None
    
    def extract_entities_pipeline(self, text: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract entities using pipeline (simple approach)
        """
        if not self.ner_pipeline or not text:
            return []
        
        try:
            results = self.ner_pipeline(text)
            
            entities = []
            for entity in results:
                if entity['score'] >= confidence_threshold:
                    entities.append({
                        "entity_type": entity['entity_group'],
                        "value": entity['word'].replace("##", "").strip(),
                        "score": float(entity['score']),
                        "start": entity.get('start', 0),
                        "end": entity.get('end', 0)
                    })
            
            return entities
        except Exception as e:
            logger.error(f"Pipeline extraction failed: {e}")
            return []
    
    def extract_entities_advanced(self, text: str, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Extract entities with BIO tagging (advanced approach)
        Better for handling multi-word entities
        """
        if not self.model or not self.tokenizer or not text:
            return []
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_offsets_mapping=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
                probabilities = torch.softmax(outputs.logits, dim=2)
            
            # Extract entities with BIO tagging
            entities = []
            current_entity = None
            offset_mapping = inputs.offset_mapping[0]
            
            for i, (pred, offset) in enumerate(zip(predictions[0], offset_mapping)):
                # Skip special tokens
                if offset[0] == 0 and offset[1] == 0:
                    continue
                
                label = self.model.config.id2label[pred.item()]
                confidence = probabilities[0][i][pred].item()
                
                if label.startswith('B-'):
                    # Save previous entity
                    if current_entity and current_entity['score'] >= confidence_threshold:
                        entities.append(current_entity)
                    
                    # Start new entity
                    entity_type = label[2:]  # Remove 'B-' prefix
                    current_entity = {
                        'entity_type': entity_type,
                        'value': text[offset[0]:offset[1]],
                        'start': int(offset[0]),
                        'end': int(offset[1]),
                        'score': confidence
                    }
                
                elif label.startswith('I-') and current_entity:
                    # Continue current entity
                    entity_type = label[2:]  # Remove 'I-' prefix
                    if entity_type == current_entity['entity_type']:
                        current_entity['value'] += ' ' + text[offset[0]:offset[1]]
                        current_entity['end'] = int(offset[1])
                        current_entity['score'] = min(current_entity['score'], confidence)
                
                elif label == 'O':
                    # End current entity
                    if current_entity and current_entity['score'] >= confidence_threshold:
                        entities.append(current_entity)
                        current_entity = None
            
            # Add last entity
            if current_entity and current_entity['score'] >= confidence_threshold:
                entities.append(current_entity)
            
            # Clean up entities
            for entity in entities:
                entity['value'] = entity['value'].replace('##', '').strip()
            
            return entities
        
        except Exception as e:
            logger.error(f"Advanced extraction failed: {e}")
            return []
    
    def merge_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Merge entities by type
        Resume-specific model should have labels like: NAME, DESIGNATION, SKILLS, EDUCATION, etc.
        """
        merged = {
            "NAME": [],
            "DESIGNATION": [],
            "SKILLS": [],
            "EDUCATION": [],
            "COLLEGE_NAME": [],
            "DEGREE": [],
            "GRADUATION_YEAR": [],
            "COMPANIES_WORKED_AT": [],
            "LOCATION": [],
            "EMAIL": [],
            "PHONE": [],
            "YEARS_EXPERIENCE": [],
        }
        
        for entity in entities:
            entity_type = entity['entity_type'].upper()
            value = entity['value'].strip()
            
            if not value:
                continue
            
            # Map to standard types
            if entity_type in merged:
                if value not in merged[entity_type]:
                    merged[entity_type].append(value)
            elif 'SKILL' in entity_type:
                if value not in merged['SKILLS']:
                    merged['SKILLS'].append(value)
            elif 'COMPANY' in entity_type or 'ORG' in entity_type:
                if value not in merged['COMPANIES_WORKED_AT']:
                    merged['COMPANIES_WORKED_AT'].append(value)
            elif 'LOC' in entity_type:
                if value not in merged['LOCATION']:
                    merged['LOCATION'].append(value)
        
        return merged
    
    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract phone and email - regex fallback"""
        contact = {"phone": [], "email": None}
        
        # Phone
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
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
    
    def extract_skills_regex(self, text: str) -> List[str]:
        """Regex fallback for skills"""
        skills = set()
        
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|Scala)\b',
            r'\b(React|Angular|Vue|Django|Flask|FastAPI|Spring|Express|Node\.js|Laravel)\b',
            r'\b(PyTorch|TensorFlow|Keras|scikit-learn|XGBoost|BERT|GPT)\b',
            r'\b(AWS|Azure|GCP|Google\s*Cloud|Docker|Kubernetes)\b',
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch)\b',
            r'\b(Git|Jira|Agile|Scrum|CI/CD)\b',
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        return sorted(list(skills))
    
    def parse_resume(self, resume_text: str, use_advanced: bool = True) -> Dict[str, Any]:
        """
        Complete parsing using resume-specific NER
        
        Args:
            resume_text: Raw resume text
            use_advanced: Use advanced BIO tagging (True) or simple pipeline (False)
        """
        if not resume_text:
            return self._empty_result("Empty resume text")
        
        try:
            # Extract entities
            if use_advanced:
                entities = self.extract_entities_advanced(resume_text, confidence_threshold=0.3)
            else:
                entities = self.extract_entities_pipeline(resume_text, confidence_threshold=0.5)
            
            logger.info(f"Extracted {len(entities)} entities from resume")
            
            # Merge entities by type
            merged = self.merge_entities(entities)
            
            # Extract with regex fallback
            contact_info = self.extract_contact_info(resume_text)
            skills_regex = self.extract_skills_regex(resume_text)
            
            # Build result
            name = merged["NAME"][0] if merged["NAME"] else "Unknown"
            email = merged["EMAIL"][0] if merged["EMAIL"] else (contact_info.get("email") or "unknown@email.com")
            phone = merged["PHONE"] or contact_info["phone"]
            
            # Skills: combine NER + regex
            skills_ner = merged["SKILLS"]
            all_skills = list(set(skills_ner + skills_regex))
            
            # Education
            education = []
            if merged["DEGREE"] or merged["COLLEGE_NAME"]:
                for i in range(max(len(merged["DEGREE"]), len(merged["COLLEGE_NAME"]))):
                    degree = merged["DEGREE"][i] if i < len(merged["DEGREE"]) else "Unknown"
                    college = merged["COLLEGE_NAME"][i] if i < len(merged["COLLEGE_NAME"]) else "Unknown"
                    year = merged["GRADUATION_YEAR"][i] if i < len(merged["GRADUATION_YEAR"]) else "Unknown"
                    education.append({
                        "degree": degree,
                        "institution": college,
                        "year": year
                    })
            
            # Experience
            years_exp = 0
            if merged["YEARS_EXPERIENCE"]:
                try:
                    years_exp = int(re.search(r'\d+', merged["YEARS_EXPERIENCE"][0]).group(0))
                except:
                    pass
            
            return {
                "name": name,
                "email_address": email,
                "contact_number": phone,
                "primary_skills": all_skills[:15] if all_skills else [],
                "secondary_skills": all_skills[15:30] if len(all_skills) > 15 else [],
                "education": [e["degree"] for e in education] if education else [],
                "total_experience_(months)": years_exp * 12,
                "current_company_name": merged["COMPANIES_WORKED_AT"][0] if merged["COMPANIES_WORKED_AT"] else "Unknown",
                "current_location": merged["LOCATION"][0] if merged["LOCATION"] else "Unknown",
                "designation": merged["DESIGNATION"][:3] if merged["DESIGNATION"] else [],
                "relevant_experience_(primary)": {"job_history": []},
                "relevant_experience_(secondary)": {"projects": [], "certifications": []},
                "applicant_description": resume_text[:200] + "...",
                "entities": entities,
                "extraction_status": "SUCCESS",
                "model_type": f"Resume-NER-V2 ({'Advanced' if use_advanced else 'Pipeline'})"
            }
        
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(str(e))
    
    def simple_read(self, resume_text: str) -> Dict[str, Any]:
        """Lightweight read - same as generic NER"""
        if not resume_text:
            return {"error": "Empty resume text", "status": "FAILED"}
        
        try:
            import hashlib
            resume_id = hashlib.md5(resume_text.encode()).hexdigest()[:12]
            
            contact_info = self.extract_contact_info(resume_text)
            
            # Quick name extraction with NER
            entities = self.extract_entities_pipeline(resume_text[:500], confidence_threshold=0.5)
            name = "Unknown"
            for e in entities:
                if e['entity_type'] == 'NAME':
                    name = e['value']
                    break
            
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