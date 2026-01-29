"""
Feature 4: Job-Resume Matching
Computes match scores and identifies skill gaps
Uses hybrid fuzzy + semantic matching for skills
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class JobResumeMatcher:
    def __init__(self):
        self._fuzzy_available = False
        self._semantic_model = None
        self._semantic_available = False
        
        # Try to import fuzzy matching
        try:
            from rapidfuzz import fuzz, process
            self._fuzz = fuzz
            self._process = process
            self._fuzzy_available = True
            logger.info("Fuzzy matching (rapidfuzz) loaded successfully")
        except ImportError:
            logger.warning("rapidfuzz not available - install with: pip install rapidfuzz")
        
        # Semantic model loaded lazily on first use
        logger.info("Semantic matching will be loaded on demand")
    
    def _load_semantic_model(self):
        """Lazy load semantic similarity model"""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer, util
                logger.info("Loading semantic model: all-MiniLM-L6-v2...")
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self._semantic_util = util
                self._semantic_available = True
                logger.info("Semantic model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self._semantic_available = False
        return self._semantic_available
    
    def _fuzzy_match_skills(self, req_skill: str, resume_skills: List[str], threshold: int = 85) -> Tuple[bool, str]:
        """
        Try to match a required skill using fuzzy string matching
        
        Returns: (matched, best_matching_skill)
        """
        if not self._fuzzy_available or not resume_skills:
            return False, ""
        
        try:
            # Find best match using token_set_ratio (handles word order, case)
            result = self._process.extractOne(
                req_skill,
                resume_skills,
                scorer=self._fuzz.token_set_ratio
            )
            
            if result and result[1] >= threshold:
                return True, result[0]
        except Exception as e:
            logger.error(f"Fuzzy matching error: {e}")
        
        return False, ""
    
    def _semantic_match_skills(self, req_skill: str, resume_skills: List[str], threshold: float = 0.75) -> Tuple[bool, str]:
        """
        Try to match a required skill using semantic similarity
        
        Returns: (matched, best_matching_skill)
        """
        if not self._load_semantic_model() or not resume_skills:
            return False, ""
        
        try:
            # Encode the required skill
            req_embedding = self._semantic_model.encode([req_skill], convert_to_tensor=True)
            
            # Encode all resume skills
            res_embeddings = self._semantic_model.encode(resume_skills, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = self._semantic_util.cos_sim(req_embedding, res_embeddings)[0]
            
            # Find best match
            max_sim_idx = similarities.argmax().item()
            max_sim = similarities[max_sim_idx].item()
            
            if max_sim >= threshold:
                return True, resume_skills[max_sim_idx]
        except Exception as e:
            logger.error(f"Semantic matching error: {e}")
        
        return False, ""
    
    def _hybrid_skill_match(self, resume_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Hybrid matching: Fuzzy first (fast), then semantic (smart)
        
        Returns: (matched_required, matched_resume, skill_gaps)
        """
        if not required_skills:
            return [], [], []
        
        matched_required = []  # Required skills that were matched
        matched_resume = []    # Resume skills that matched
        skill_gaps = []        # Required skills not found
        
        for req_skill in required_skills:
            # Try fuzzy match first (fast)
            fuzzy_matched, fuzzy_skill = self._fuzzy_match_skills(req_skill, resume_skills, threshold=85)
            
            if fuzzy_matched:
                matched_required.append(req_skill)
                matched_resume.append(fuzzy_skill)
                continue
            
            # Fall back to semantic match (slower but smarter)
            semantic_matched, semantic_skill = self._semantic_match_skills(req_skill, resume_skills, threshold=0.75)
            
            if semantic_matched:
                matched_required.append(req_skill)
                matched_resume.append(semantic_skill)
                continue
            
            # No match found
            skill_gaps.append(req_skill)
        
        return matched_required, matched_resume, skill_gaps
    
    def compute_skill_match(self, resume_skills: List[str], required_skills: List[str]) -> float:
        """Jaccard similarity for skills (legacy method)"""
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
        
        # Ensure inputs are dicts
        if not isinstance(resume_data, dict):
            resume_data = {}
        if not isinstance(jd_data, dict):
            jd_data = {"required_skills": [], "required_experience": 0, "required_education": ""}
        
        # Extract resume fields (handle both formats)
        if 'skills' in resume_data:
            # Get all skills
            resume_skills = set(resume_data.get('skills', []))
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
        
        # Skill Match (Jaccard similarity) - CASE INSENSITIVE + SUBSTRING MATCHING
        if required_skills:
            resume_skills_lower = {s.lower().strip() for s in resume_skills if s}
            required_skills_lower = {s.lower().strip() for s in required_skills if s}
            
            # Find matches using substring matching (more flexible)
            matched = set()
            for req_skill in required_skills_lower:
                for res_skill in resume_skills_lower:
                    # Check both directions: exact match or substring
                    if req_skill == res_skill or req_skill in res_skill or res_skill in req_skill:
                        matched.add(req_skill)
                        break
            
            union = resume_skills_lower.union(required_skills_lower)
            
            skill_match = (len(matched) / len(required_skills_lower)) * 100 if required_skills_lower else 0
            
            # Find original case for matched and gaps
            matched_skills = []
            for s in resume_skills:
                s_lower = s.lower().strip()
                for req_skill in required_skills_lower:
                    if req_skill == s_lower or req_skill in s_lower or s_lower in req_skill:
                        matched_skills.append(s)
                        break
            
            skill_gaps = []
            for s in required_skills:
                s_lower = s.lower().strip()
                if s_lower not in matched:
                    skill_gaps.append(s)
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