"""
Hiring Predictor Service
Loads trained XGBoost model and makes real-time hiring predictions.
"""

import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class HiringPredictor:
    """Predicts hiring outcomes based on candidate answer features."""
    
    def __init__(self):
        """Initialize the hiring predictor."""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_loaded = False
        
    def load_model(self, model_dir: Optional[str] = None) -> bool:
        """
        Load the trained model, scaler, and feature names from disk.
        
        Args:
            model_dir: Directory containing model files. If None, uses default location.
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_dir is None:
                # Get absolute path to models directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                model_dir = os.path.join(project_root, 'models', 'hiring_predictor')
            
            # Load model
            model_path = os.path.join(model_dir, 'xgboost_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            features_path = os.path.join(model_dir, 'feature_names.pkl')
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully from {model_dir}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please train the model first using Train_Hiring_Predictor.ipynb")
            self.model_loaded = False
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def predict(self, ml_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a hiring prediction based on candidate features.
        
        Args:
            ml_features: Dictionary containing ML features from answer analysis
            
        Returns:
            Dictionary containing prediction results with:
                - hire_probability: Float between 0-1
                - recommendation: String recommendation
                - confidence: String confidence level
                - recommendation_text: Detailed explanation
                - top_features: List of most important features
        """
        if not self.model_loaded:
            # Attempt to load model if not already loaded
            if not self.load_model():
                return self._get_default_prediction()
        
        try:
            # Extract features in correct order
            feature_values = []
            for feature_name in self.feature_names:
                value = ml_features.get(feature_name, 0)
                feature_values.append(value)
            
            # Convert to numpy array and reshape
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            hire_probability = float(self.model.predict_proba(X_scaled)[0][1])
            
            # Get recommendation based on probability
            recommendation, confidence, rec_text = self._get_recommendation(
                hire_probability, ml_features
            )
            
            # Get top contributing features
            top_features = self._get_top_features(X_scaled[0], ml_features)
            
            return {
                'hire_probability': hire_probability,
                'recommendation': recommendation,
                'confidence': confidence,
                'recommendation_text': rec_text,
                'top_features': top_features,
                'model_version': '1.0'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self._get_default_prediction()
    
    def _get_recommendation(
        self, probability: float, features: Dict[str, Any]
    ) -> Tuple[str, str, str]:
        """
        Generate recommendation based on hire probability and features.
        
        Returns:
            Tuple of (recommendation, confidence, recommendation_text)
        """
        quality_score = features.get('overall_quality_score', 0)
        
        # Determine recommendation
        if probability >= 0.70:
            recommendation = 'strongly_recommend_hire'
            confidence = 'high'
            rec_text = (
                f"Strong hire recommendation with {probability*100:.1f}% confidence. "
                f"Candidate demonstrates excellent answer quality (score: {quality_score:.0f}/100) "
                f"with strong technical knowledge and communication skills."
            )
        elif probability >= 0.55:
            recommendation = 'recommend_hire'
            confidence = 'moderate'
            rec_text = (
                f"Positive hiring recommendation with {probability*100:.1f}% confidence. "
                f"Candidate shows good potential (quality score: {quality_score:.0f}/100). "
                f"Consider for interview round."
            )
        elif probability >= 0.45:
            recommendation = 'borderline'
            confidence = 'low'
            rec_text = (
                f"Borderline case with {probability*100:.1f}% hire probability. "
                f"Candidate quality score is {quality_score:.0f}/100. "
                f"Recommend additional assessment or screening interview."
            )
        elif probability >= 0.30:
            recommendation = 'recommend_reject'
            confidence = 'moderate'
            rec_text = (
                f"Not recommended for hire ({probability*100:.1f}% probability). "
                f"Answer quality score of {quality_score:.0f}/100 indicates gaps in "
                f"experience or communication. Consider for junior positions only."
            )
        else:
            recommendation = 'strongly_recommend_reject'
            confidence = 'high'
            rec_text = (
                f"Strong reject recommendation ({probability*100:.1f}% hire probability). "
                f"Low answer quality score ({quality_score:.0f}/100) suggests significant "
                f"gaps in skills or preparation."
            )
        
        return recommendation, confidence, rec_text
    
    def _get_top_features(
        self, scaled_features: np.ndarray, original_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify the most important features contributing to the prediction.
        
        Returns:
            List of top 5 features with their values and importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []
        
        # Get feature importances from the model
        importances = self.model.feature_importances_
        
        # Create list of (feature_name, importance, value) tuples
        feature_info = []
        for i, feature_name in enumerate(self.feature_names):
            importance = importances[i]
            value = original_features.get(feature_name, 0)
            feature_info.append({
                'name': feature_name,
                'importance': float(importance),
                'value': float(value)
            })
        
        # Sort by importance and return top 5
        feature_info.sort(key=lambda x: x['importance'], reverse=True)
        return feature_info[:5]
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """
        Return a default prediction when model is not available.
        """
        return {
            'hire_probability': 0.5,
            'recommendation': 'manual_review',
            'confidence': 'none',
            'recommendation_text': (
                'Model not available. Please train the model using '
                'Train_Hiring_Predictor.ipynb before making predictions.'
            ),
            'top_features': [],
            'model_version': 'none'
        }


# Singleton instance
_hiring_predictor_instance: Optional[HiringPredictor] = None


def get_hiring_predictor() -> HiringPredictor:
    """Get or create the singleton HiringPredictor instance."""
    global _hiring_predictor_instance
    if _hiring_predictor_instance is None:
        _hiring_predictor_instance = HiringPredictor()
        _hiring_predictor_instance.load_model()  # Attempt to load model on initialization
    return _hiring_predictor_instance
