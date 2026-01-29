"""
Verify ML System Integration
Quick check to ensure all components are properly connected.
"""

import sys

def check_imports():
    """Verify all imports work"""
    print("🔍 Checking imports...")
    
    try:
        from backend.services.answer_analyzer import get_answer_analyzer
        print("  ✅ answer_analyzer imported")
    except Exception as e:
        print(f"  ❌ answer_analyzer failed: {e}")
        return False
    
    try:
        from backend.services.hiring_predictor import get_hiring_predictor
        print("  ✅ hiring_predictor imported")
    except Exception as e:
        print(f"  ❌ hiring_predictor failed: {e}")
        return False
    
    return True


def check_services():
    """Verify services can be instantiated"""
    print("\n🔍 Checking services...")
    
    try:
        from backend.services.answer_analyzer import get_answer_analyzer
        analyzer = get_answer_analyzer()
        
        # Test feature extraction
        test_answers = [
            {
                "question": "Tell me about your experience",
                "answer": "I have 5 years of experience with Python, Django, and React. I built a microservices architecture that improved performance by 40%."
            }
        ]
        
        features = analyzer.analyze_response(test_answers)
        print(f"  ✅ Answer analyzer working (quality score: {features['overall_quality_score']:.1f})")
        
    except Exception as e:
        print(f"  ❌ Answer analyzer failed: {e}")
        return False
    
    try:
        from backend.services.hiring_predictor import get_hiring_predictor
        predictor = get_hiring_predictor()
        
        if predictor.model_loaded:
            print("  ✅ Hiring predictor loaded (model found)")
        else:
            print("  ⚠️ Hiring predictor ready but model not trained yet")
        
    except Exception as e:
        print(f"  ❌ Hiring predictor failed: {e}")
        return False
    
    return True


def check_database():
    """Verify database has training data"""
    print("\n🔍 Checking database...")
    
    try:
        from pymongo import MongoClient
        
        client = MongoClient("mongodb://localhost:27017/")
        db = client['resume_screening_db']
        
        responses = db['questionnaire_responses'].count_documents({})
        analytics = db['question_analytics'].count_documents({})
        
        print(f"  ✅ Database connected")
        print(f"  📊 Responses: {responses}")
        print(f"  📊 Analytics: {analytics}")
        
        # Check training data
        training_responses = db['questionnaire_responses'].count_documents({
            "response_id": {"$regex": "^train_"}
        })
        
        training_analytics = db['question_analytics'].count_documents({
            "response_id": {"$regex": "^train_"},
            "outcome": {"$exists": True}
        })
        
        print(f"  📊 Training samples: {training_responses} responses, {training_analytics} labeled")
        
        if training_analytics >= 10:
            print("  ✅ Sufficient training data available")
        else:
            print("  ⚠️ Need more training data (run populate_training_data.py)")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database check failed: {e}")
        return False


def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    
    import os
    
    model_dir = "models/hiring_predictor"
    required_files = [
        "xgboost_model.pkl",
        "scaler.pkl",
        "feature_names.pkl"
    ]
    
    if not os.path.exists(model_dir):
        print(f"  ⚠️ Model directory not found: {model_dir}")
        print("  📝 Run Train_Hiring_Predictor.ipynb to train model")
        return False
    
    all_exist = True
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} missing")
            all_exist = False
    
    if not all_exist:
        print("  📝 Run Train_Hiring_Predictor.ipynb to train model")
    else:
        print("  ✅ All model files present")
    
    return all_exist


def main():
    """Run all checks"""
    print("=" * 60)
    print("🚀 ML Hiring Prediction System - Integration Check")
    print("=" * 60)
    
    checks = {
        "Imports": check_imports(),
        "Services": check_services(),
        "Database": check_database(),
        "Model Files": check_model_files()
    }
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {check_name}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 All checks passed! System is ready.")
        print("📝 Next step: Run Train_Hiring_Predictor.ipynb if model not trained")
    else:
        print("\n⚠️ Some checks failed. Review errors above.")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
