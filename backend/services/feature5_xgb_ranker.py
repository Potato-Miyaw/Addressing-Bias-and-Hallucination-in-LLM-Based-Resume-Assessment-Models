"""
Feature 5: XGBoost Fairness-Aware Ranking
Uses XGBoost + Fairlearn ExponentiatedGradient for bias mitigation
Based on validated notebook approach
"""

import xgboost as xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import joblib
import os

class FairnessAwareRanker:
    def __init__(self, model_path: str = "/home/claude/models_saved/xgb_fairness_model.pkl"):
        self.model_path = model_path
        self.baseline_model = None
        self.fairness_model = None
        
    def engineer_features(self, resume_data: Dict, jd_data: Dict, match_data: Dict) -> Dict[str, float]:
        """
        Engineer features from resume, JD, and match data
        
        10 features (from notebook):
        1. skills_count
        2. education_level (ordinal)
        3. certification_count
        4. experience_years
        5. projects_count
        6. ai_score (match score)
        7. skills_match_score
        8. resume_keywords_count
        9. jd_keywords_count
        10. tfidf_similarity (using match score as proxy)
        """
        
        # Ensure all inputs are dicts
        if not isinstance(match_data, dict):
            match_data = {"match_score": 0, "skill_match": 0}
        
        if not isinstance(resume_data, dict):
            resume_data = {}
            
        if not isinstance(jd_data, dict):
            jd_data = {"required_skills": []}
        
        # Education level mapping
        edu_mapping = {
            "high school": 1,
            "associate": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4
        }
        
        # Extract resume data - handle both field name variants
        # Primary skills from BERT NER format
        skills = resume_data.get("skills", [])
        education = resume_data.get("education", [])
        
        # Handle certifications - check if it exists and is a dict
        sec_exp = resume_data.get("relevant_experience_(secondary)", {})
        if isinstance(sec_exp, dict):
            certifications = sec_exp.get("certifications", [])
        else:
            certifications = []
        
        # Experience in months from BERT NER
        exp_months = resume_data.get("total_experience_(months)", 0)
        if not isinstance(exp_months, (int, float)):
            exp_months = 0
        
        # Ensure lists are actually lists
        if not isinstance(skills, list):
            skills = []
        if not isinstance(education, list):
            education = []
        if not isinstance(certifications, list):
            certifications = []
        
        # Education level
        edu_level = 1
        if education:
            edu_item = education[0]
            edu_str = edu_item.get("degree", "") if isinstance(edu_item, dict) else str(edu_item)
            edu_str = edu_str.lower()
            for key, val in edu_mapping.items():
                if key in edu_str:
                    edu_level = val
                    break
        
        # Experience years from months
        exp_years = exp_months / 12 if exp_months > 0 else 0
        
        # Match scores with safe defaults
        try:
            skill_match = (match_data.get("skill_match", 0) or 0) / 100.0
            overall_match = (match_data.get("match_score", 0) or 0) / 100.0
        except (TypeError, ZeroDivisionError):
            skill_match = 0.0
            overall_match = 0.0
        
        features = {
            "skills_count": len(skills),
            "education_level": edu_level,
            "certification_count": len(certifications),
            "experience_years": exp_years,
            "projects_count": 0,  # Not extracted yet
            "ai_score": overall_match,
            "skills_match_score": skills_match,
            "resume_keywords_count": len(skills) + len(certifications),
            "jd_keywords_count": len(jd_data.get("required_skills", [])),
            "tfidf_similarity": overall_match  # Using match score as proxy
        }
        
        return features
    
    def train_baseline_xgboost(self, X_train, y_train):
        """
        Train baseline XGBoost with GridSearchCV
        Parameters from notebook
        """
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.baseline_model = grid_search.best_estimator_
        return self.baseline_model
    
    def train_fairness_model(self, X_train, y_train, sensitive_features, eps=0.02):
        """
        Train fairness-aware model using Fairlearn ExponentiatedGradient
        with DemographicParity constraint
        
        eps=0.02 from notebook (best balance between fairness and accuracy)
        """
        if self.baseline_model is None:
            self.train_baseline_xgboost(X_train, y_train)
        
        # Fairlearn mitigation
        mitigator = ExponentiatedGradient(
            self.baseline_model,
            constraints=DemographicParity(),
            eps=eps
        )
        
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        self.fairness_model = mitigator
        return self.fairness_model
    
    def predict(self, X, use_fairness=True):
        """Predict using baseline or fairness model"""
        model = self.fairness_model if use_fairness and self.fairness_model else self.baseline_model
        
        if model is None:
            raise ValueError("No model trained yet")
        
        return model.predict(X)
    
    def predict_proba(self, X, use_fairness=True):
        """Predict probabilities"""
        model = self.fairness_model if use_fairness and self.fairness_model else self.baseline_model
        
        if model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # Fairlearn might not have predict_proba
            preds = model.predict(X)
            return np.column_stack([1-preds, preds])
    
    def rank_candidates(self, candidates_data: List[Dict], jd_data: Dict, use_fairness=True) -> List[Dict]:
        """
        Rank candidates for a job
        
        Args:
            candidates_data: List of dicts with resume_data, match_data, demographics
            jd_data: Job description data
            use_fairness: Use fairness model if True
        
        Returns:
            Ranked list of candidates with scores
        """
        if not candidates_data:
            return []
        
        try:
            # Engineer features for all candidates
            X = []
            valid_candidates = []
            
            for candidate in candidates_data:
                try:
                    features = self.engineer_features(
                        candidate.get('resume_data', {}),
                        jd_data,
                        candidate.get('match_data', {})
                    )
                    X.append(list(features.values()))
                    valid_candidates.append(candidate)
                except Exception as e:
                    print(f"Warning: Could not engineer features for candidate: {e}")
                    continue
            
            if not valid_candidates:
                # If no valid candidates, return original with default scores
                for i, candidate in enumerate(candidates_data):
                    candidate['ranking_score'] = 0.0
                    candidate['rank'] = i + 1
                return candidates_data
            
            X = np.array(X)
            
            # Get predictions
            if self.baseline_model or self.fairness_model:
                probs = self.predict_proba(X, use_fairness=use_fairness)
                scores = probs[:, 1]  # Probability of being qualified
            else:
                # No model trained - use match scores
                scores = np.array([c.get('match_data', {}).get('match_score', 0) / 100.0 for c in valid_candidates])
            
            # Add scores and rank
            for i, candidate in enumerate(valid_candidates):
                candidate['ranking_score'] = float(scores[i])
            
            # Sort by score (descending)
            ranked = sorted(valid_candidates, key=lambda x: x['ranking_score'], reverse=True)
            
            # Add ranks
            for rank, candidate in enumerate(ranked, 1):
                candidate['rank'] = rank
            
            return ranked
        except Exception as e:
            print(f"Error in rank_candidates: {e}")
            # Return candidates with default ranking
            for i, candidate in enumerate(candidates_data):
                candidate['ranking_score'] = 0.0
                candidate['rank'] = i + 1
            return candidates_data
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        data = {
            'baseline_model': self.baseline_model,
            'fairness_model': self.fairness_model
        }
        
        joblib.dump(data, self.model_path)
    
    def load_models(self):
        """Load trained models"""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.baseline_model = data.get('baseline_model')
            self.fairness_model = data.get('fairness_model')
            return True
        return False