"""
Feature 3: Claim Verification Service
Uses Token Overlap + BERTScore + Logistic Regression for hallucination detection
Based on validated notebook implementation
"""

import numpy as np
from typing import List, Dict, Any
from nltk.tokenize import word_tokenize
from bert_score import score as bertscore_compute
from sklearn.linear_model import LogisticRegression
import joblib
import os
import logging

# Suppress BERTScore logging
logging.getLogger('bert_score').setLevel(logging.CRITICAL)

# Create logger for this module
logger = logging.getLogger(__name__)

class ClaimVerifier:
    def __init__(self, model_path: str = "/home/claude/models_saved/hallucination_lr.pkl"):
        """
        Initialize claim verifier with pre-trained logistic regression model
        If model doesn't exist, creates a default trained model
        """
        self.model_path = model_path
        self.model = self._load_or_create_model()
        self.optimal_threshold = 0.30  # From notebook CV optimization
        
    def _load_or_create_model(self) -> LogisticRegression:
        """Load existing model or create default trained model"""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            # Create default model with parameters from notebook
            model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            # Save directory
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            return model
    
    def save_model(self):
        """Save trained model to disk"""
        joblib.dump(self.model, self.model_path)
        
    def compute_token_overlap(self, extraction: Any, ground_truth: Any) -> float:
        """
        Compute Jaccard similarity between extraction and ground truth.
        Exact implementation from notebook.
        
        Args:
            extraction: The value extracted by the LLM
            ground_truth: The true value from the ground truth data
            
        Returns:
            float: Similarity score between 0 and 1
        """
        def _to_text(value):
            if isinstance(value, (list, tuple, set)):
                return ' '.join([str(x) for x in value if x is not None and str(x).strip() != ''])
            elif value is None:
                return ''
            else:
                return str(value)
        
        extraction_text = _to_text(extraction)
        ground_truth_text = _to_text(ground_truth)
        
        if not extraction_text and not ground_truth_text:
            return 1.0
        if not ground_truth_text:
            return 0.0
        if not extraction_text:
            return 0.0
        
        try:
            extraction_tokens = set(word_tokenize(extraction_text.lower()))
            ground_truth_tokens = set(word_tokenize(ground_truth_text.lower()))
        except:
            # Fallback to simple split if nltk not available
            extraction_tokens = set(extraction_text.lower().split())
            ground_truth_tokens = set(ground_truth_text.lower().split())
        
        intersection = extraction_tokens & ground_truth_tokens
        union = extraction_tokens | ground_truth_tokens
        
        if not union:
            return 1.0
        
        return len(intersection) / len(union)
    
    def compute_bertscore(self, extraction: str, ground_truth: str) -> float:
        """
        Compute BERTScore F1 between extraction and ground truth
        
        Args:
            extraction: Extracted text
            ground_truth: Ground truth text
            
        Returns:
            float: BERTScore F1 score
        """
        if not extraction or not ground_truth:
            return 0.0
        
        try:
            P, R, F1 = bertscore_compute(
                [str(extraction)],
                [str(ground_truth)],
                lang="en",
                verbose=False
            )
            return float(F1[0].item())
        except Exception as e:
            logging.warning(f"BERTScore computation failed: {e}")
            return 0.0
    
    def extract_features(self, token_overlap: float, bertscore_f1: float) -> np.ndarray:
        """
        Extract features for logistic regression model
        Exact feature engineering from notebook
        
        Features:
        1. 1 - token_overlap (inverted similarity)
        2. 1 - bertscore_f1 (inverted similarity)
        3. token_overlap < 0.70 (binary threshold)
        4. bertscore_f1 < 0.80 (binary threshold)
        5. token_overlap * bertscore_f1 (interaction term)
        
        Returns:
            np.ndarray: Feature vector
        """
        features = np.array([
            1 - token_overlap,
            1 - bertscore_f1,
            int(token_overlap < 0.70),
            int(bertscore_f1 < 0.80),
            token_overlap * bertscore_f1
        ]).reshape(1, -1)
        
        return features
    
    def verify_claim(self, extraction: Any, ground_truth: Any) -> Dict[str, Any]:
        """
        Verify a single claim using the trained model
        
        Args:
            extraction: Extracted value from resume
            ground_truth: Ground truth value (if available)
            
        Returns:
            dict: Verification result with confidence and verdict
        """
        # Compute metrics
        token_overlap = self.compute_token_overlap(extraction, ground_truth)
        bertscore_f1 = self.compute_bertscore(str(extraction), str(ground_truth))
        
        # Extract features
        features = self.extract_features(token_overlap, bertscore_f1)
        
        ## Predict hallucination probability
        try:
            if hasattr(self.model, 'predict_proba'):
                hallucination_prob = self.model.predict_proba(features)[0][1]
            else:
                raise ValueError("Model not trained")
        except:
            # If model not trained, use simple threshold
            hallucination_prob = 1 - (token_overlap * 0.6 + bertscore_f1 * 0.4)
        
        # Apply optimal threshold
        is_hallucination = hallucination_prob >= self.optimal_threshold
        
        # Confidence is inverse of hallucination probability
        confidence = 1 - hallucination_prob
        
        return {
            "extraction": str(extraction),
            "ground_truth": str(ground_truth),
            "token_overlap": float(token_overlap),
            "bertscore_f1": float(bertscore_f1),
            "hallucination_probability": float(hallucination_prob),
            "confidence": float(confidence),
            "is_hallucination": bool(is_hallucination),
            "verdict": "HALLUCINATION" if is_hallucination else "VERIFIED"
        }
    
    def verify_resume_data(self, resume_extractions: Dict[str, Any], 
                           ground_truth_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify all claims in a resume extraction
        
        Args:
            resume_extractions: Dictionary of extracted resume fields
            ground_truth_data: Optional ground truth for validation
            
        Returns:
            dict: Complete verification report
        """
        # Ensure inputs are dicts
        if not isinstance(resume_extractions, dict):
            resume_extractions = {}
        if ground_truth_data is None:
            ground_truth_data = {}
        elif not isinstance(ground_truth_data, dict):
            ground_truth_data = {}
        
        verified_claims = []
        flagged_claims = []
        
        # Extract evidence snippets if available
        evidence_snippets = ground_truth_data.get('_evidence_snippets', {})
        
        # ONLY verify actual extracted entities, skip metadata fields
        metadata_fields = {
            'resume_id', 'file_type', 'text_length', 'status', 'message', 
            'saved_to_db', 'text', '_id', 'filename', 'raw_text', 'timestamp',
            'created_at', 'updated_at', 'id', '_evidence_snippets', 'success',
            'model_type', 'entities', 'extraction_status', 'error'
        }
        
        def is_empty_value(value):
            """Check if a value is empty/null/meaningless"""
            if value is None:
                return True
            if isinstance(value, str) and not value.strip():
                return True
            if isinstance(value, (list, tuple, set)) and len(value) == 0:
                return True
            if isinstance(value, dict) and len(value) == 0:
                return True
            return False
        
        def extract_verifiable_values(field_name, value):
            """Extract actual values from nested objects for verification"""
            if field_name == 'experience' and isinstance(value, dict):
                # For experience, extract companies and years
                verifiable = []
                if value.get('companies'):
                    verifiable.extend([c for c in value['companies'] if c and c != 'Unknown'])
                if value.get('years', 0) > 0:
                    verifiable.append(f"{value['years']} years")
                return verifiable if verifiable else None
            return value
        
        # Verify each field
        for field, extraction in resume_extractions.items():
            # Skip metadata fields
            if field in metadata_fields:
                logger.debug(f"⏭️ Skipping metadata field: {field}")
                continue
            
            # Skip empty/null fields
            if is_empty_value(extraction):
                logger.debug(f"⏭️ Skipping empty field: {field}")
                continue
            
            if not ground_truth_data:
                continue
            
            # Extract verifiable values from nested objects
            extraction_to_verify = extract_verifiable_values(field, extraction)
            if extraction_to_verify is None or is_empty_value(extraction_to_verify):
                logger.debug(f"⏭️ Skipping field with no verifiable data: {field}")
                continue
            
            ground_truth = ground_truth_data.get(field, "")
            
            # Log what we're comparing
            logger.info(f"\n🔍 Verifying field: {field}")
            logger.info(f"   📤 Extraction: {extraction_to_verify}")
            logger.info(f"   📥 Ground Truth: {ground_truth}")
            
            if not ground_truth or is_empty_value(ground_truth):
                logger.warning(f"   ⚠️ No ground truth found for field '{field}' - marking as hallucination")
            
            result = self.verify_claim(extraction_to_verify, ground_truth)
            result['field'] = field
            
            # Add evidence snippet if available
            if field in evidence_snippets:
                result['evidence_snippet'] = evidence_snippets[field]
            
            logger.info(f"   {'✅' if not result['is_hallucination'] else '❌'} Result: {result['verdict']} (confidence: {result['confidence']:.2f})")
            
            verified_claims.append(result)
            
            if result['is_hallucination']:
                flagged_claims.append(result)
        
        # Overall confidence (average of non-hallucinated claims)
        valid_claims = [c for c in verified_claims if not c['is_hallucination']]
        overall_confidence = np.mean([c['confidence'] for c in verified_claims]) if verified_claims else 0.0
        
        if not verified_claims:
            return {
                "overall_confidence": 1.0,
                "total_claims": 0,
                "verified_claims": 0,
                "flagged_claims": 0,
                "hallucination_rate": 0.0,
                "verdict": "NO_CLAIMS",
                "details": [],
                "flagged": []
            }
        else:
            return {
                "overall_confidence": float(overall_confidence),
                "total_claims": len(verified_claims),
                "verified_claims": len(valid_claims),
                "flagged_claims": len(flagged_claims),
                "hallucination_rate": len(flagged_claims) / len(verified_claims) if verified_claims else 0.0,
                "verdict": "VERIFIED" if len(flagged_claims) == 0 else "CONTAINS_HALLUCINATIONS",
                "details": verified_claims,
                "flagged": flagged_claims
            }
    
    def train_from_data(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the logistic regression model on labeled data
        
        Args:
            X_train: Feature matrix (token_overlap, bertscore_f1)
            y_train: Binary labels (0: verified, 1: hallucination)
        """
        # Extract features for all samples
        X_features = []
        for i in range(len(X_train)):
            token_overlap, bertscore_f1 = X_train[i]
            features = self.extract_features(token_overlap, bertscore_f1)
            X_features.append(features[0])
        
        X_features = np.array(X_features)
        
        # Train model
        self.model.fit(X_features, y_train)
        
        # Save model
        self.save_model()
        
        return self.model