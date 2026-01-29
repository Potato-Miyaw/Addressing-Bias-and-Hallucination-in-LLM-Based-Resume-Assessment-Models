"""
Populate Training Data for Hiring Predictor
Generates sample labeled responses for initial model training.
"""

from pymongo import MongoClient
from datetime import datetime
import uuid


def populate_training_data():
    """Insert sample training data with labels for model training."""
    
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client['resume_screening_db']
    responses_collection = db['questionnaire_responses']
    analytics_collection = db['question_analytics']
    
    print("🔗 Connected to MongoDB")
    
    # Sample training data: 5 hired + 6 rejected = 11 samples
    training_samples = [
        # HIRED CANDIDATES (high quality scores)
        {
            "response_id": f"train_hired_{i}",
            "candidate_name": f"Strong Candidate {i}",
            "candidate_email": f"strong{i}@example.com",
            "position": "Software Engineer",
            "submitted_at": datetime.utcnow(),
            "answers": [
                {"question": "Tell us about your experience", "answer": "Extensive experience with multiple technologies"},
                {"question": "Why this role?", "answer": "Perfect alignment with my career goals and technical skills"}
            ],
            "ml_features": {
                "overall_quality_score": 60 + (i * 8),  # 60, 68, 76, 84, 92
                "avg_word_count": 120 + (i * 10),
                "avg_completeness": 85 + i * 2,
                "avg_tech_density": 8.0 + i * 0.5,
                "avg_structure_score": 75 + i * 3,
                "avg_specificity_score": 70 + i * 4,
                "avg_sentiment": 0.4 + i * 0.05,
                "total_tech_keywords": 15 + i * 2,
                "total_action_verbs": 8 + i,
                "pct_with_examples": 70 + i * 5,
                "pct_with_metrics": 60 + i * 5,
                "pct_with_actions": 80 + i * 3,
                "consistency_score": 75 + i * 3
            },
            "outcome": "hired"
        }
        for i in range(1, 6)
    ] + [
        # REJECTED CANDIDATES (low quality scores)
        {
            "response_id": f"train_rejected_{i}",
            "candidate_name": f"Weak Candidate {i}",
            "candidate_email": f"weak{i}@example.com",
            "position": "Software Engineer",
            "submitted_at": datetime.utcnow(),
            "answers": [
                {"question": "Tell us about your experience", "answer": "Some experience"},
                {"question": "Why this role?", "answer": "Need a job"}
            ],
            "ml_features": {
                "overall_quality_score": 22 + (i * 3),  # 22, 25, 28, 31, 34, 37
                "avg_word_count": 25 + (i * 5),
                "avg_completeness": 30 + i * 3,
                "avg_tech_density": 1.0 + i * 0.3,
                "avg_structure_score": 25 + i * 2,
                "avg_specificity_score": 20 + i * 3,
                "avg_sentiment": 0.0 + i * 0.02,
                "total_tech_keywords": 2 + i,
                "total_action_verbs": 1 + (i // 2),
                "pct_with_examples": 10 + i * 3,
                "pct_with_metrics": 5 + i * 2,
                "pct_with_actions": 20 + i * 4,
                "consistency_score": 40 + i * 2
            },
            "outcome": "rejected"
        }
        for i in range(1, 7)
    ]
    
    # Insert responses and outcomes
    inserted_count = 0
    for sample in training_samples:
        response_id = sample["response_id"]
        outcome = sample.pop("outcome")
        
        # Insert response (or update if exists)
        responses_collection.update_one(
            {"response_id": response_id},
            {"$set": sample},
            upsert=True
        )
        
        # Insert outcome in analytics
        analytics_collection.update_one(
            {"response_id": response_id},
            {
                "$set": {
                    "response_id": response_id,
                    "outcome": outcome,
                    "question_ratings": [],
                    "submitted_at": datetime.utcnow()
                }
            },
            upsert=True
        )
        
        inserted_count += 1
        status = "✅" if outcome == "hired" else "❌"
        quality = sample["ml_features"]["overall_quality_score"]
        print(f"{status} {response_id}: quality={quality:.0f}, outcome={outcome}")
    
    print(f"\n🎉 Successfully populated {inserted_count} training samples!")
    print(f"   - Hired: 5 candidates (quality scores: 60-92)")
    print(f"   - Rejected: 6 candidates (quality scores: 22-37)")
    print(f"\n📝 Next step: Run Train_Hiring_Predictor.ipynb to train the model")
    
    client.close()


if __name__ == "__main__":
    populate_training_data()
