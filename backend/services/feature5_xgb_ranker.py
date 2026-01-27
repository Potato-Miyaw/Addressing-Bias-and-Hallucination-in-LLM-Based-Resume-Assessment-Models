"""
Feature 5: XGBoost Fairness-Aware Ranking
Uses XGBoost + Fairlearn ExponentiatedGradient for bias mitigation
Based on validated notebook approach
"""

import os
from typing import List, Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV

class FairnessAwareRanker:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.baseline_model = None
        self.fairness_model = None
        if self.model_path is None:
            self.model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models",
                "xgb_fairness_model.pkl"
            )
        self.load_models()

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r"[a-zA-Z0-9\\+\\#\\.]+", str(text).lower())

    def _normalize_list(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [v.strip() for v in str(value).split(",") if v.strip()]

    def _extract_resume_fields(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        skills = (
            resume_data.get("skills")
            or resume_data.get("primary_skills")
            or []
        )
        if isinstance(skills, list):
            skills_list = skills
        else:
            skills_list = self._normalize_list(skills)

        secondary_skills = resume_data.get("secondary_skills", [])
        if isinstance(secondary_skills, list):
            skills_list = skills_list + secondary_skills

        certifications = resume_data.get("certifications", [])
        certifications_list = certifications if isinstance(certifications, list) else self._normalize_list(certifications)

        education = resume_data.get("education", [])
        education_list = education if isinstance(education, list) else [education]

        experience_years = 0.0
        if isinstance(resume_data.get("experience"), dict):
            experience_years = float(resume_data["experience"].get("years", 0) or 0)
        elif resume_data.get("total_experience_(months)") is not None:
            experience_years = float(resume_data.get("total_experience_(months)", 0)) / 12.0
        elif resume_data.get("experience_years") is not None:
            experience_years = float(resume_data.get("experience_years", 0) or 0)

        return {
            "skills_list": skills_list,
            "certifications_list": certifications_list,
            "education_list": education_list,
            "experience_years": experience_years
        }

    def _education_level(self, education_list: List[Any]) -> int:
        edu_mapping = {
            "high school": 1,
            "associate": 1,
            "bachelor": 2,
            "bachelors": 2,
            "b.sc": 2,
            "bsc": 2,
            "b.tech": 2,
            "btech": 2,
            "master": 3,
            "masters": 3,
            "mba": 3,
            "m.tech": 3,
            "mtech": 3,
            "phd": 4,
            "doctorate": 4
        }
        level = 0
        for edu in education_list:
            text = str(edu).lower()
            for key, value in edu_mapping.items():
                if key in text:
                    level = max(level, value)
        return level

    def _skills_match_score(self, resume_skills: List[str], jd_skills: List[str]) -> float:
        resume_set = {s.lower().strip() for s in resume_skills if str(s).strip()}
        jd_set = {s.lower().strip() for s in jd_skills if str(s).strip()}
        if not jd_set:
            return 0.0
        return len(resume_set.intersection(jd_set)) / len(jd_set)

    def engineer_features_batch(
        self,
        candidates_data: List[Dict[str, Any]],
        jd_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
        """
        Engineer features from resume, JD, and match data (batch).
        
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
        jd_skills = jd_data.get("required_skills", []) or []
        jd_text = jd_data.get("jd_text")
        if not jd_text:
            jd_text = " ".join([str(s) for s in jd_skills])

        resume_texts = []
        for candidate in candidates_data:
            resume_data = candidate.get("resume_data", {})
            extracted = self._extract_resume_fields(resume_data)
            resume_texts.append(" ".join(extracted["skills_list"]))

        tfidf_similarity = [0.0] * len(candidates_data)
        if resume_texts:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            corpus = resume_texts + [jd_text]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            resume_vecs = tfidf_matrix[:len(resume_texts)]
            jd_vec = tfidf_matrix[len(resume_texts):]
            if jd_vec.shape[0] == 1:
                tfidf_similarity = cosine_similarity(resume_vecs, jd_vec).reshape(-1).tolist()

        features_list = []
        for idx, candidate in enumerate(candidates_data):
            resume_data = candidate.get("resume_data", {})
            match_data = candidate.get("match_data", {}) or {}
            extracted = self._extract_resume_fields(resume_data)

            skills_list = extracted["skills_list"]
            certifications_list = extracted["certifications_list"]

            edu_level = self._education_level(extracted["education_list"])
            exp_years = extracted["experience_years"]

            skills_match = match_data.get("skill_match")
            if skills_match is None:
                skills_match = self._skills_match_score(skills_list, jd_skills) * 100.0
            skills_match_score = float(skills_match) / 100.0

            overall_match = match_data.get("match_score")
            if overall_match is None:
                overall_match = skills_match
            ai_score = float(overall_match) / 100.0

            resume_keywords = self._tokenize(" ".join(skills_list))
            jd_keywords = self._tokenize(jd_text)

            features = {
                "skills_count": len(skills_list),
                "education_level": edu_level,
                "certification_count": len(certifications_list),
                "experience_years": exp_years,
                "projects_count": 0,
                "ai_score": ai_score,
                "skills_match_score": skills_match_score,
                "resume_keywords_count": len(resume_keywords),
                "jd_keywords_count": len(jd_keywords),
                "tfidf_similarity": float(tfidf_similarity[idx]) if idx < len(tfidf_similarity) else 0.0
            }
            features_list.append(features)

        feature_names = list(features_list[0].keys()) if features_list else []
        X = np.array([[f[name] for name in feature_names] for f in features_list]) if features_list else np.array([])
        return X, feature_names, features_list
    
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
        
        # Engineer features for all candidates
        X, _, _ = self.engineer_features_batch(candidates_data, jd_data)
        
        # Get predictions
        if self.baseline_model or self.fairness_model:
            probs = self.predict_proba(X, use_fairness=use_fairness)
            scores = probs[:, 1]  # Probability of being qualified
        else:
            # No model trained - use match scores
            scores = np.array([c.get("match_data", {}).get("match_score", 0) / 100.0 for c in candidates_data])
        
        # Add scores and rank
        for i, candidate in enumerate(candidates_data):
            candidate['ranking_score'] = float(scores[i])
        
        # Sort by score (descending)
        ranked = sorted(candidates_data, key=lambda x: x['ranking_score'], reverse=True)
        
        # Add ranks
        for rank, candidate in enumerate(ranked, 1):
            candidate['rank'] = rank
        
        return ranked

    def rank_candidates_with_metrics(
        self,
        candidates_data: List[Dict[str, Any]],
        jd_data: Dict[str, Any],
        use_fairness: bool = True,
        sensitive_attribute: str = "gender",
        hire_threshold: float = 0.5
    ) -> Dict[str, Any]:
        ranked = self.rank_candidates(candidates_data, jd_data, use_fairness=use_fairness)

        for candidate in ranked:
            candidate["hire_probability"] = float(candidate.get("ranking_score", 0.0))
            match_score = candidate.get("match_data", {}).get("match_score")
            if match_score is None:
                candidate["match_score"] = round(candidate["hire_probability"], 3)
            else:
                match_score_val = float(match_score)
                candidate["match_score"] = round(match_score_val / 100.0, 4) if match_score_val > 1 else round(match_score_val, 4)

            verification = candidate.get("verification")
            if isinstance(verification, dict):
                candidate["verification_status"] = verification.get("verdict", "UNKNOWN")
            else:
                candidate["verification_status"] = candidate.get("verification_status", "UNKNOWN")

            candidate["decision"] = "HIRE" if candidate["hire_probability"] >= hire_threshold else "REJECT"

        fairness_metrics = self.compute_fairness_metrics(
            ranked,
            sensitive_attribute=sensitive_attribute,
            hire_threshold=hire_threshold
        )

        fairness_metrics["fairness_mode"] = "ON" if use_fairness else "OFF"

        return {
            "ranked_candidates": ranked,
            "fairness_metrics": fairness_metrics
        }

    def compute_fairness_metrics(
        self,
        candidates_data: List[Dict[str, Any]],
        sensitive_attribute: str = "gender",
        hire_threshold: float = 0.5
    ) -> Dict[str, Any]:
        if not candidates_data:
            return {
                "impact_ratio": None,
                "demographic_parity": None,
                "equal_opportunity": None,
                "selection_rates": {},
                "total_candidates": 0
            }

        sensitive_values = []
        decisions = []
        y_true = []
        has_labels = True

        for candidate in candidates_data:
            demographics = candidate.get("demographics", {}) or {}
            sensitive_value = demographics.get(sensitive_attribute)
            if sensitive_value is None:
                sensitive_value = demographics.get("race_gender", "Unknown")
            sensitive_values.append(str(sensitive_value))

            hire_prob = candidate.get("hire_probability", candidate.get("ranking_score", 0.0))
            decisions.append(1 if hire_prob >= hire_threshold else 0)

            label = candidate.get("label")
            if label is None:
                label = candidate.get("y")
            if label is None and isinstance(candidate.get("resume_data"), dict):
                label = candidate["resume_data"].get("y")
            if label is None:
                has_labels = False
            else:
                y_true.append(int(label))

        selection_rates = {}
        for group in sorted(set(sensitive_values)):
            idx = [i for i, v in enumerate(sensitive_values) if v == group]
            if not idx:
                continue
            rate = sum(decisions[i] for i in idx) / float(len(idx))
            selection_rates[group] = round(rate, 4)

        impact_ratio = None
        demographic_parity = None
        note = None
        if selection_rates:
            rates = list(selection_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate > 0:
                impact_ratio = round(min_rate / max_rate, 4)
            demographic_parity = round(max_rate - min_rate, 4)
            if len(selection_rates) <= 1:
                impact_ratio = 1.0
                demographic_parity = 0.0
                note = "Only one sensitive group detected; fairness metrics are limited."

        equal_opportunity = None
        if has_labels and y_true:
            try:
                from fairlearn.metrics import equal_opportunity_difference
                equal_opportunity = float(equal_opportunity_difference(
                    y_true,
                    decisions,
                    sensitive_features=sensitive_values
                ))
                equal_opportunity = round(equal_opportunity, 4)
            except Exception:
                equal_opportunity = None

        return {
            "impact_ratio": impact_ratio,
            "demographic_parity": demographic_parity,
            "equal_opportunity": equal_opportunity,
            "selection_rates": selection_rates,
            "total_candidates": len(candidates_data),
            "note": note
        }
    
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
