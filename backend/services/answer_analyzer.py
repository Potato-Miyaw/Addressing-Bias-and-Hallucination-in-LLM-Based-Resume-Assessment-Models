"""
Answer Analyzer Service
Extracts ML features from candidate questionnaire responses for hiring prediction.
"""

import re
from typing import Dict, List, Any, Optional
from textblob import TextBlob


class AnswerAnalyzer:
    """Analyzes candidate answers and extracts features for ML prediction."""
    
    def __init__(self):
        """Initialize the answer analyzer with technical keywords."""
        self.tech_keywords = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust',
            'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'spring', 'express',
            'nodejs', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn',
            'jquery', 'bootstrap', 'tailwind',
            
            # Databases
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'cassandra',
            'dynamodb', 'oracle', 'sqlite', 'mariadb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
            'terraform', 'ansible', 'ci/cd', 'devops',
            
            # Concepts & Methodologies
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum', 'tdd', 'bdd',
            'oop', 'solid', 'design patterns', 'algorithms', 'data structures',
            'machine learning', 'deep learning', 'ai', 'nlp', 'computer vision',
            
            # Tools
            'git', 'vscode', 'intellij', 'eclipse', 'postman', 'jira', 'confluence',
            'slack', 'teams', 'figma', 'sketch'
        }
        
        self.action_verbs = {
            'implemented', 'developed', 'created', 'built', 'designed', 'architected',
            'deployed', 'optimized', 'improved', 'reduced', 'increased', 'led',
            'managed', 'coordinated', 'collaborated', 'achieved', 'delivered',
            'migrated', 'refactored', 'automated', 'integrated', 'configured'
        }
        
    def analyze_answer(self, answer: str, question: str = "") -> Dict[str, Any]:
        """
        Analyze a single answer and extract features.
        
        Args:
            answer: The candidate's answer text
            question: The question text (optional, for context)
            
        Returns:
            Dictionary containing answer features
        """
        if not answer or not isinstance(answer, str):
            return self._get_empty_features()
            
        answer_lower = answer.lower()
        
        # Basic metrics
        word_count = len(answer.split())
        char_count = len(answer)
        sentence_count = len([s for s in answer.split('.') if s.strip()])
        
        # Completeness score (0-100)
        completeness = min(100, (word_count / 50) * 100)  # 50 words = 100%
        
        # Technical keyword density
        tech_keywords_found = sum(1 for keyword in self.tech_keywords 
                                   if keyword in answer_lower)
        tech_density = (tech_keywords_found / max(word_count, 1)) * 100
        
        # Check for examples (indicators: "for example", "such as", "like when", etc.)
        has_examples = any(phrase in answer_lower for phrase in [
            'for example', 'for instance', 'such as', 'like when',
            'one time', 'once i', 'i remember', 'specifically'
        ])
        
        # Check for metrics/numbers (indicates quantifiable achievements)
        has_metrics = bool(re.search(r'\d+\s*%|\d+\s*users|\d+\s*customers|'
                                      r'\d+\s*projects|\d+\s*hours|\d+\s*days|'
                                      r'\d+\s*weeks|\d+\s*months|\$\d+', 
                                      answer_lower))
        
        # Check for action verbs (shows proactive behavior)
        action_verbs_found = sum(1 for verb in self.action_verbs 
                                  if verb in answer_lower)
        has_action_verbs = action_verbs_found > 0
        
        # Sentiment analysis
        try:
            blob = TextBlob(answer)
            sentiment_polarity = blob.sentiment.polarity  # -1 to 1
            sentiment_subjectivity = blob.sentiment.subjectivity  # 0 to 1
        except Exception:
            sentiment_polarity = 0.0
            sentiment_subjectivity = 0.5
            
        # Structure score (presence of multiple sentences, paragraphs)
        paragraph_count = len([p for p in answer.split('\n\n') if p.strip()])
        structure_score = min(100, (paragraph_count * 25 + sentence_count * 10))
        
        # Specificity score (combination of examples, metrics, technical terms)
        specificity_indicators = sum([
            has_examples * 30,
            has_metrics * 30,
            min(tech_keywords_found * 10, 40)
        ])
        specificity_score = min(100, specificity_indicators)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'completeness': completeness,
            'tech_keywords_found': tech_keywords_found,
            'tech_density': tech_density,
            'has_examples': has_examples,
            'has_metrics': has_metrics,
            'has_action_verbs': has_action_verbs,
            'action_verbs_count': action_verbs_found,
            'sentiment_polarity': sentiment_polarity,
            'sentiment_subjectivity': sentiment_subjectivity,
            'structure_score': structure_score,
            'specificity_score': specificity_score
        }
    
    def analyze_response(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze all answers in a questionnaire response and compute aggregate features.
        
        Args:
            answers: List of answer dictionaries with 'question' and 'answer' fields
            
        Returns:
            Dictionary containing aggregate ML features
        """
        if not answers:
            return self._get_empty_aggregate_features()
            
        # Analyze each answer
        analyzed_answers = []
        for ans in answers:
            answer_text = ans.get('answer', '')
            question_text = ans.get('question', '')
            features = self.analyze_answer(answer_text, question_text)
            analyzed_answers.append(features)
        
        if not analyzed_answers:
            return self._get_empty_aggregate_features()
        
        # Compute aggregate statistics
        total_answers = len(analyzed_answers)
        
        # Average metrics
        avg_word_count = sum(a['word_count'] for a in analyzed_answers) / total_answers
        avg_completeness = sum(a['completeness'] for a in analyzed_answers) / total_answers
        avg_tech_density = sum(a['tech_density'] for a in analyzed_answers) / total_answers
        avg_structure_score = sum(a['structure_score'] for a in analyzed_answers) / total_answers
        avg_specificity_score = sum(a['specificity_score'] for a in analyzed_answers) / total_answers
        avg_sentiment = sum(a['sentiment_polarity'] for a in analyzed_answers) / total_answers
        
        # Total counts
        total_tech_keywords = sum(a['tech_keywords_found'] for a in analyzed_answers)
        total_action_verbs = sum(a['action_verbs_count'] for a in analyzed_answers)
        
        # Percentages
        pct_with_examples = sum(1 for a in analyzed_answers if a['has_examples']) / total_answers * 100
        pct_with_metrics = sum(1 for a in analyzed_answers if a['has_metrics']) / total_answers * 100
        pct_with_actions = sum(1 for a in analyzed_answers if a['has_action_verbs']) / total_answers * 100
        
        # Consistency metrics (standard deviation)
        import statistics
        word_counts = [a['word_count'] for a in analyzed_answers]
        consistency_score = 100 - min(100, statistics.stdev(word_counts) if len(word_counts) > 1 else 0)
        
        # Overall quality score (0-100)
        overall_quality = self._calculate_overall_quality(
            avg_completeness, avg_tech_density, avg_structure_score,
            avg_specificity_score, pct_with_examples, pct_with_metrics,
            pct_with_actions, avg_sentiment
        )
        
        return {
            # Summary metrics
            'total_answers': total_answers,
            'overall_quality_score': overall_quality,
            
            # Average metrics
            'avg_word_count': avg_word_count,
            'avg_completeness': avg_completeness,
            'avg_tech_density': avg_tech_density,
            'avg_structure_score': avg_structure_score,
            'avg_specificity_score': avg_specificity_score,
            'avg_sentiment': avg_sentiment,
            
            # Total counts
            'total_tech_keywords': total_tech_keywords,
            'total_action_verbs': total_action_verbs,
            
            # Percentages
            'pct_with_examples': pct_with_examples,
            'pct_with_metrics': pct_with_metrics,
            'pct_with_actions': pct_with_actions,
            
            # Quality indicators
            'consistency_score': consistency_score,
            
            # Individual answer features (for detailed analysis)
            'individual_answers': analyzed_answers
        }
    
    def _calculate_overall_quality(
        self, avg_completeness: float, avg_tech_density: float,
        avg_structure_score: float, avg_specificity_score: float,
        pct_with_examples: float, pct_with_metrics: float,
        pct_with_actions: float, avg_sentiment: float
    ) -> float:
        """
        Calculate overall quality score using weighted combination of features.
        
        Returns:
            Overall quality score (0-100)
        """
        # Weights for different aspects
        weights = {
            'completeness': 0.20,      # 20% - sufficient detail
            'tech_density': 0.15,      # 15% - technical knowledge
            'structure': 0.10,         # 10% - well-organized
            'specificity': 0.20,       # 20% - concrete examples
            'examples': 0.15,          # 15% - storytelling
            'metrics': 0.10,           # 10% - quantifiable results
            'actions': 0.05,           # 5% - proactive language
            'sentiment': 0.05          # 5% - positive tone
        }
        
        # Normalize sentiment from [-1, 1] to [0, 100]
        sentiment_normalized = (avg_sentiment + 1) * 50
        
        # Calculate weighted score
        quality_score = (
            avg_completeness * weights['completeness'] +
            min(avg_tech_density * 10, 100) * weights['tech_density'] +
            avg_structure_score * weights['structure'] +
            avg_specificity_score * weights['specificity'] +
            pct_with_examples * weights['examples'] +
            pct_with_metrics * weights['metrics'] +
            pct_with_actions * weights['actions'] +
            sentiment_normalized * weights['sentiment']
        )
        
        return min(100, max(0, quality_score))
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty features for invalid answers."""
        return {
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'paragraph_count': 0,
            'completeness': 0,
            'tech_keywords_found': 0,
            'tech_density': 0,
            'has_examples': False,
            'has_metrics': False,
            'has_action_verbs': False,
            'action_verbs_count': 0,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.5,
            'structure_score': 0,
            'specificity_score': 0
        }
    
    def _get_empty_aggregate_features(self) -> Dict[str, Any]:
        """Return empty aggregate features."""
        return {
            'total_answers': 0,
            'overall_quality_score': 0,
            'avg_word_count': 0,
            'avg_completeness': 0,
            'avg_tech_density': 0,
            'avg_structure_score': 0,
            'avg_specificity_score': 0,
            'avg_sentiment': 0,
            'total_tech_keywords': 0,
            'total_action_verbs': 0,
            'pct_with_examples': 0,
            'pct_with_metrics': 0,
            'pct_with_actions': 0,
            'consistency_score': 0,
            'individual_answers': []
        }


# Singleton instance
_answer_analyzer_instance: Optional[AnswerAnalyzer] = None


def get_answer_analyzer() -> AnswerAnalyzer:
    """Get or create the singleton AnswerAnalyzer instance."""
    global _answer_analyzer_instance
    if _answer_analyzer_instance is None:
        _answer_analyzer_instance = AnswerAnalyzer()
    return _answer_analyzer_instance
