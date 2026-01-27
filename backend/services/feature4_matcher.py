"""
Feature 4: Job-Resume Matching
Computes match scores and identifies skill gaps
"""

from typing import Dict, Any, List
import numpy as np

class JobResumeMatcher:
    def __init__(self):
        pass
    
    def compute_skill_match(self, resume_skills: List[str], required_skills: List[str]) -> float:
        """Jaccard similarity for skills"""
        if not required_skills:
            return 1.0
        
        resume_set = set([s.lower() for s in resume_skills])
        required_set = set([s.lower() for s in required_skills])
        
        intersection = resume_set & required_set
        union = resume_set | required_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def compute_experience_match(self, resume_exp: int, required_exp: int) -> float:
        """Experience match score"""
        if required_exp == 0:
            return 1.0
        if resume_exp >= required_exp:
            return 1.0
        return resume_exp / required_exp
    
    def compute_education_match(self, resume_edu: str, required_edu: str) -> float:
        """Education level match"""
        edu_hierarchy = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5
        }
        
        resume_level = 0
        required_level = 0
        
        for key, val in edu_hierarchy.items():
            if key in resume_edu.lower():
                resume_level = val
            if key in required_edu.lower():
                required_level = val
        
        if resume_level >= required_level:
            return 1.0
        return resume_level / required_level if required_level > 0 else 0.5
    
    def match_resume_to_job(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, float]:
        """Match resume to job description"""
        
        # Extract resume fields (handle both formats)
        if 'primary_skills' in resume_data:
            # New comprehensive format - COMBINE primary + secondary
            resume_skills = set(
                resume_data.get('primary_skills', []) + 
                resume_data.get('secondary_skills', [])
            )
            resume_exp_months = resume_data.get('total_experience_(months)', 0)
            resume_exp_years = resume_exp_months / 12
            resume_education = resume_data.get('education', [])
        else:
            # Old simple format
            resume_skills = set(resume_data.get('skills', []))
            exp_dict = resume_data.get('experience', {})
            resume_exp_years = exp_dict.get('years', 0) if isinstance(exp_dict, dict) else 0
            resume_education = resume_data.get('education', [])
        
        # Extract JD requirements
        required_skills = set(jd_data.get('required_skills', []))
        required_exp = jd_data.get('required_experience', 0)
        required_edu = jd_data.get('required_education', '')
        
        # Skill Match (Jaccard similarity) - CASE INSENSITIVE
        if required_skills:
            resume_skills_lower = {s.lower().strip() for s in resume_skills}
            required_skills_lower = {s.lower().strip() for s in required_skills}
            
            matched = resume_skills_lower.intersection(required_skills_lower)
            union = resume_skills_lower.union(required_skills_lower)
            
            skill_match = (len(matched) / len(union)) * 100 if union else 0
            
            # Find original case for matched and gaps
            matched_skills = [s for s in resume_skills if s.lower() in matched]
            skill_gaps = [s for s in required_skills if s.lower() not in resume_skills_lower]
        else:
            skill_match = 100.0
            matched_skills = []
            skill_gaps = []
        
        # Experience Match
        if required_exp > 0:
            experience_match = min((resume_exp_years / required_exp) * 100, 100)
        else:
            experience_match = 100.0
        
        # Education Match (ordinal)
        edu_hierarchy = {
            'high school': 1,
            'associate': 2,
            'bachelor': 3,
            'master': 4,
            'phd': 5,
            'doctorate': 5
        }
        
        # Get education levels
        resume_edu_level = 0
        for edu in resume_education:
            edu_str = str(edu).lower() if not isinstance(edu, dict) else str(edu.get('degree', '')).lower()
            for key, value in edu_hierarchy.items():
                if key in edu_str:
                    resume_edu_level = max(resume_edu_level, value)
        
        required_edu_level = 0
        required_edu_lower = required_edu.lower()
        for key, value in edu_hierarchy.items():
            if key in required_edu_lower:
                required_edu_level = value
                break
        
        if required_edu_level > 0:
            education_match = min((resume_edu_level / required_edu_level) * 100, 100)
        else:
            education_match = 100.0
        
        # Overall Match (weighted average)
        match_score = (
            skill_match * 0.5 +
            experience_match * 0.3 +
            education_match * 0.2
        )
        
        return {
            "match_score": round(match_score, 1),
            "skill_match": round(skill_match, 1),
            "experience_match": round(experience_match, 1),
            "education_match": round(education_match, 1),
            "skill_gaps": skill_gaps,
            "matched_skills": matched_skills,
            "fulfillment_percentage": round(match_score, 1)
        }