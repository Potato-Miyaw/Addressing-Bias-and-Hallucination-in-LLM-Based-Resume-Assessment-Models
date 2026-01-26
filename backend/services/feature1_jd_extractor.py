"""
Feature 1: Job Description Extraction using TinyLlama (non-gated alternative to Gemma)
Extracts: skills, experience, education, certifications from JD text
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from typing import Dict, Any

class JDExtractor:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize TinyLlama for JD extraction (non-gated, free to use)"""
        try:
            print(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.use_llm = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            print("Falling back to regex-based extraction")
            self.use_llm = False
    
    def extract_jd_data(self, jd_text: str) -> Dict[str, Any]:
        """Extract structured data from job description"""
        
        if not self.use_llm:
            return self._fallback_extraction(jd_text)
        
        # Simpler, more direct prompt
        prompt = f"""Extract job requirements from this description. Return ONLY a JSON object, no other text.

    Job Description:
    {jd_text}

    Return this exact format:
    {{"required_skills": ["Python", "AWS"], "required_experience": 5, "required_education": "Bachelor's", "certifications": ["AWS Certified"]}}

    JSON:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,  # Greedy decoding for consistency
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract ONLY the JSON part (after "JSON:")
            if "JSON:" in response:
                json_part = response.split("JSON:")[-1].strip()
            else:
                json_part = response
            
            # Find first { and last }
            start = json_part.find('{')
            end = json_part.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = json_part[start:end]
                data = json.loads(json_str)
                return {
                    "required_skills": data.get("required_skills", []),
                    "required_experience": data.get("required_experience", 0),
                    "required_education": data.get("required_education", ""),
                    "certifications": data.get("certifications", []),
                    "status": "SUCCESS"
                }
        except Exception as e:
            print(f"LLM extraction failed: {e}")
        
        # Always fallback if LLM fails
        return self._fallback_extraction(jd_text)
    
    def _fallback_extraction(self, jd_text: str) -> Dict[str, Any]:
        """Regex-based fallback extraction (always works)"""
        
        # Skills - expanded patterns
        skill_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|R|MATLAB|Scala)\b',
            r'\b(React|Angular|Vue|Django|Flask|FastAPI|Spring|Node\.js|Express|Laravel)\b',
            r'\b(Docker|Kubernetes|AWS|Azure|GCP|Git|Jenkins|CI/CD|Terraform|Ansible)\b',
            r'\b(SQL|PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Cassandra|Oracle|DynamoDB)\b',
            r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|TensorFlow|PyTorch|Scikit-learn|Keras)\b',
            r'\b(HTML|CSS|REST|GraphQL|Microservices|Agile|Scrum|DevOps)\b'
        ]
        
        skills = set()
        for pattern in skill_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            skills.update([m.strip() for m in matches])
        
        # Experience
        exp_match = re.search(r'(\d+)\+?\s*(?:to\s+\d+)?\s*years?', jd_text, re.IGNORECASE)
        experience = int(exp_match.group(1)) if exp_match else 0
        
        # Education
        education = "Bachelor's"
        if re.search(r'Master|MS|MSc|M\.S|M\.Sc', jd_text, re.IGNORECASE):
            education = "Master's"
        elif re.search(r'PhD|Ph\.D|Doctorate', jd_text, re.IGNORECASE):
            education = "PhD"
        elif re.search(r'High School|Diploma', jd_text, re.IGNORECASE):
            education = "High School"
        
        # Certifications
        cert_patterns = [
            r'(AWS Certified[\w\s]+)',
            r'(Azure[\w\s]+Certified)',
            r'(Google Cloud[\w\s]+Certified)',
            r'\b(PMP|CISSP|CEH|CISA|CompTIA\s+\w+)\b',
            r'(Certified[\w\s]+(?:Professional|Specialist|Expert|Developer|Administrator))'
        ]
        
        certifications = []
        for pattern in cert_patterns:
            matches = re.findall(pattern, jd_text, re.IGNORECASE)
            certifications.extend([m.strip() if isinstance(m, str) else m[0].strip() for m in matches])
        
        return {
            "required_skills": list(skills),
            "required_experience": experience,
            "required_education": education,
            "certifications": list(set(certifications)),
            "status": "FALLBACK"
        }