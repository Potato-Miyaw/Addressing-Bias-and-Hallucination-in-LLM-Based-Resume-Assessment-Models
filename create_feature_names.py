"""
Quick fix: Create feature_names.pkl for the model
"""

import pickle
import os

# Feature names in the correct order
feature_names = [
    'overall_quality_score',
    'avg_word_count',
    'avg_completeness',
    'avg_tech_density',
    'avg_structure_score',
    'avg_specificity_score',
    'avg_sentiment',
    'total_tech_keywords',
    'total_action_verbs',
    'pct_with_examples',
    'pct_with_metrics',
    'pct_with_actions',
    'consistency_score'
]

# Save to models directory
model_dir = 'models/hiring_predictor'
os.makedirs(model_dir, exist_ok=True)

features_path = os.path.join(model_dir, 'feature_names.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(feature_names, f)

print(f"✅ Created {features_path}")
print(f"📊 Features: {', '.join(feature_names)}")
