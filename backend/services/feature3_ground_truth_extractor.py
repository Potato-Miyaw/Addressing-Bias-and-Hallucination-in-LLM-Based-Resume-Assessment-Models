"""
Ground Truth Extractor
Extracts actual values from raw resume text to use as ground truth for verification
Checks if LLM extractions are actually present in the resume (prevents hallucinations)
"""

import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class GroundTruthExtractor:
    def __init__(self):
        """Initialize ground truth extractor"""
        pass
    
    def extract_ground_truth_from_text(self, resume_text: str, resume_extractions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract ground truth values from raw resume text
        
        Strategy: For each extracted field, search the resume text to verify it's actually there
        This prevents hallucinations - we only accept extractions that appear in the original text
        
        Args:
            resume_text: Raw resume text
            resume_extractions: LLM-extracted data
            
        Returns:
            dict: Ground truth data extracted from raw text
        """
        ground_truth = {}
        
        # Normalize text for searching
        text_lower = resume_text.lower()
        
        # 1. NAME - Check if extracted name appears in first 500 chars
        if 'name' in resume_extractions:
            extracted_name = resume_extractions['name']
            if extracted_name and extracted_name != "Unknown":
                # Check if name appears in text
                name_lower = extracted_name.lower()
                if name_lower in text_lower[:500]:  # Name should be in first 500 chars
                    ground_truth['name'] = extracted_name
                else:
                    # Try to find the actual name in first few lines
                    lines = resume_text.split('\n')[:5]
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 3 and not any(skip in line.lower() for skip in ['email', '@', 'phone', 'resume']):
                            # Simple name pattern
                            if re.match(r'^[A-Z][a-zA-Z\s\-\'\.]+$', line) and 2 <= len(line.split()) <= 4:
                                ground_truth['name'] = line
                                break
        
        # 2. EMAIL - Extract from text
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, resume_text)
        if email_match:
            ground_truth['email_address'] = email_match.group(0)
        
        # 3. PHONE - Extract from text
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, resume_text)
            if phone_match:
                ground_truth['contact_number'] = [phone_match.group(0)]
                break
        
        # 4. SKILLS - Check if extracted skills actually appear in text
        if 'primary_skills' in resume_extractions:
            verified_skills = []
            extracted_skills = resume_extractions.get('primary_skills', []) + resume_extractions.get('secondary_skills', [])
            
            for skill in extracted_skills:
                skill_lower = skill.lower()
                # Check if skill appears in text (case-insensitive)
                if skill_lower in text_lower:
                    verified_skills.append(skill)
            
            ground_truth['primary_skills'] = verified_skills
        
        # 5. EDUCATION - Check if extracted education appears in text
        if 'education' in resume_extractions:
            verified_education = []
            extracted_education = resume_extractions.get('education', [])
            
            for edu in extracted_education:
                edu_text = str(edu).lower()
                # Check if degree/institution appears in text
                if any(word in text_lower for word in edu_text.split()[:3]):  # Check first 3 words
                    verified_education.append(edu)
            
            ground_truth['education'] = verified_education
        
        # 6. EXPERIENCE - Extract years from text
        years_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+(?:in|as|working)',
        ]
        
        for pattern in years_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                ground_truth['total_experience_(months)'] = years * 12
                break
        
        # 7. COMPANIES - Extract company names mentioned in text
        # Look for common job-related patterns
        company_patterns = [
            r'(?:at|@)\s+([A-Z][A-Za-z\s&\.]+?)(?:\s+\(|\s+-|\s+•|\n)',
            r'(?:Company|Employer|Organization)\s*:\s*([A-Z][A-Za-z\s&\.]+)',
        ]
        
        companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, resume_text)
            for match in matches:
                company = match.strip()
                if 3 < len(company) < 50:  # Reasonable company name length
                    companies.append(company)
        
        if companies:
            ground_truth['current_company_name'] = companies[0]
            ground_truth['companies_worked_at'] = list(set(companies))[:5]
        
        # 8. LOCATION - Extract location
        location_patterns = [
            r'(?:Location|Address|City)\s*:\s*([A-Z][A-Za-z\s,]+)',
            r'\b([A-Z][a-z]+,\s*[A-Z]{2})\b',  # City, ST
            r'(?:based in|located in|residing in)\s+([A-Z][a-zA-Z\s,]+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, resume_text)
            if match:
                ground_truth['current_location'] = match.group(1).strip()
                break
        
        # 9. CERTIFICATIONS - Check if extracted certs appear in text
        if 'relevant_experience_(secondary)' in resume_extractions:
            secondary_exp = resume_extractions['relevant_experience_(secondary)']
            if isinstance(secondary_exp, dict) and 'certifications' in secondary_exp:
                extracted_certs = secondary_exp['certifications']
                
                verified_certs = []
                for cert in extracted_certs:
                    cert_text = str(cert).lower() if not isinstance(cert, dict) else str(cert.get('certificate_title', '')).lower()
                    # Check if cert appears in text
                    if any(word in text_lower for word in cert_text.split()[:3]):
                        verified_certs.append(cert)
                
                if verified_certs:
                    ground_truth['certifications'] = verified_certs
        
        # 10. DESIGNATION/JOB TITLES - Extract from text
        title_patterns = [
            r'\b(Senior|Lead|Principal|Staff|Chief)\s+(Engineer|Developer|Scientist|Architect|Manager|Analyst)\b',
            r'\b(Software|Machine\s+Learning|Data|Backend|Frontend|Full\s*Stack|DevOps)\s+(Engineer|Developer)\b',
        ]
        
        titles = []
        for pattern in title_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    titles.append(' '.join(match))
                else:
                    titles.append(match)
        
        if titles:
            ground_truth['designation'] = list(set(titles))[:3]
        
        logger.info(f"Extracted ground truth: {len(ground_truth)} fields verified from text")
        
        return ground_truth
    
    def compare_extraction_to_text(self, field: str, extraction: Any, ground_truth: Any, resume_text: str) -> Dict[str, Any]:
        """
        Compare a specific extraction to ground truth
        
        Args:
            field: Field name
            extraction: Extracted value
            ground_truth: Ground truth value
            resume_text: Raw resume text
            
        Returns:
            dict: Comparison result
        """
        result = {
            "field": field,
            "extraction": extraction,
            "ground_truth": ground_truth,
            "verified": False,
            "reason": ""
        }
        
        # Convert to comparable formats
        extraction_str = str(extraction).lower()
        ground_truth_str = str(ground_truth).lower()
        text_lower = resume_text.lower()
        
        # Check 1: Does extraction match ground truth?
        if extraction_str == ground_truth_str:
            result["verified"] = True
            result["reason"] = "Exact match with ground truth"
            return result
        
        # Check 2: Does extraction appear in original text?
        if extraction_str in text_lower:
            result["verified"] = True
            result["reason"] = "Appears in resume text"
            return result
        
        # Check 3: Are they similar? (for lists/arrays)
        if isinstance(extraction, list) and isinstance(ground_truth, list):
            extraction_set = set(str(x).lower() for x in extraction)
            ground_truth_set = set(str(x).lower() for x in ground_truth)
            
            intersection = extraction_set & ground_truth_set
            if len(intersection) > 0:
                similarity = len(intersection) / len(extraction_set | ground_truth_set)
                if similarity > 0.5:
                    result["verified"] = True
                    result["reason"] = f"High similarity ({similarity:.2f})"
                    return result
        
        # Check 4: Partial match?
        if len(extraction_str) > 5 and extraction_str in ground_truth_str:
            result["verified"] = True
            result["reason"] = "Partial match"
            return result
        
        # Not verified
        result["reason"] = "Does not match ground truth or appear in text"
        return result