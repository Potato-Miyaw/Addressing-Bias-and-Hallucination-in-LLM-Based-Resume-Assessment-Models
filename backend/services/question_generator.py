"""
Question Generator Service - Hybrid approach using SmolLM2-1.7B-Instruct + Templates
Generates personalized questionnaires for candidates based on match results
"""

import hashlib
import logging
from typing import Dict, List, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Global model instance (lazy loading)
_model = None
_tokenizer = None
_model_loaded = False


def _load_model():
    """Lazy load SmolLM2 model"""
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded:
        return
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading SmolLM2-1.7B-Instruct on {device}...")
        _tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        _model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        _model_loaded = True
        logger.info("✅ SmolLM2 model loaded successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to load SmolLM2 model: {e}. Will use templates only.")
        _model_loaded = False


def generate_template_questions(
    job_title: str,
    skill_gaps: List[str],
    matched_skills: List[str],
    experience_years: int,
    required_experience: int
) -> List[Dict[str, Any]]:
    """
    Generate rule-based template questions (60% of questionnaire)
    Fast, reliable, and more contextual
    
    Args:
        job_title: Job title
        skill_gaps: Skills the candidate is missing
        matched_skills: Skills the candidate has
        experience_years: Candidate's years of experience
        required_experience: Required years of experience
    
    Returns:
        List of question dictionaries
    """
    questions = []
    question_id = 1
    
    # 1. Skill Gap Deep Dive (Top 2 critical skills only - more focused)
    for i, skill in enumerate(skill_gaps[:2]):
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "multiple_choice",
            "text": f"What best describes your current experience with {skill}?",
            "options": [
                f"Never used {skill}, but eager to learn",
                f"Basic knowledge through courses/tutorials",
                f"Used {skill} in 1-2 small projects",
                f"Regular use in professional projects (6+ months)",
                f"Expert level with multiple years of production experience"
            ],
            "required": True,
            "category": "technical_skills",
            "skill": skill
        })
        question_id += 1
        
        # Add scenario-based follow-up
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"Describe a specific technical challenge you've faced that required skills similar to {skill}. How did you approach it?",
            "required": True,
            "category": "technical_skills",
            "skill": skill,
            "placeholder": "Share a real example with context, your approach, and the outcome..."
        })
        question_id += 1
    
    # 2. Experience & Growth Trajectory
    if experience_years < required_experience:
        gap_years = required_experience - experience_years
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"This {job_title} role typically requires {required_experience} years of experience. With your {experience_years} years, describe 2-3 accomplishments that demonstrate you can perform at this level.",
            "required": True,
            "category": "experience",
            "placeholder": "Focus on measurable impact, technical complexity, or leadership responsibilities..."
        })
        question_id += 1
    
    # 3. Matched Skills - Depth Check (Pick 1 strong skill)
    if matched_skills:
        top_skill = matched_skills[0]
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"You mentioned {top_skill} as a strength. Walk us through your most impactful project using {top_skill}. What was the business outcome?",
            "required": True,
            "category": "project_experience",
            "skill": top_skill,
            "placeholder": "Include: Problem context, your technical approach, tools/frameworks used, measurable results..."
        })
        question_id += 1
    
    # 4. Role-Specific Scenarios (contextual based on job title)
    role_keywords = job_title.lower()
    
    if any(keyword in role_keywords for keyword in ['engineer', 'developer', 'architect']):
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"Describe your approach to technical decision-making. Share an example where you chose between multiple architectural or implementation options.",
            "required": True,
            "category": "problem_solving",
            "placeholder": "What were the options? What criteria did you use? What was the outcome?"
        })
        question_id += 1
    
    if any(keyword in role_keywords for keyword in ['senior', 'lead', 'principal', 'manager']):
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"Tell us about a time you mentored someone or helped a team member overcome a technical challenge.",
            "required": True,
            "category": "leadership",
            "placeholder": "What was the situation? How did you help? What was the result for them and the team?"
        })
        question_id += 1
    
    if any(keyword in role_keywords for keyword in ['machine learning', 'ml', 'ai', 'data scientist']):
        questions.append({
            "question_id": f"Q{question_id}",
            "type": "text",
            "text": f"Describe an ML model you deployed to production. What challenges did you face with model performance, drift, or monitoring?",
            "required": True,
            "category": "technical_depth",
            "placeholder": "Include model type, deployment strategy, monitoring approach, and how you handled issues..."
        })
        question_id += 1
    
    # 5. Logistics (always include these)
    questions.append({
        "question_id": f"Q{question_id}",
        "type": "multiple_choice",
        "text": "When are you available to start?",
        "options": ["Immediately", "Within 2 weeks", "Within 1 month", "More than 1 month"],
        "required": True,
        "category": "logistics"
    })
    question_id += 1
    
    questions.append({
        "question_id": f"Q{question_id}",
        "type": "multiple_choice",
        "text": "What is your preferred work arrangement?",
        "options": ["Remote", "Hybrid", "On-site", "Flexible"],
        "required": True,
        "category": "logistics"
    })
    
    return questions


def generate_ai_questions(
    job_title: str,
    skill_gaps: List[str],
    matched_skills: List[str],
    experience_years: int,
    num_questions: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate AI-powered behavioral/scenario questions using SmolLM2 (40% of questionnaire)
    More contextual, intelligent, and role-specific
    
    Args:
        job_title: Job title
        skill_gaps: Skills the candidate is missing
        matched_skills: Skills the candidate has
        experience_years: Candidate's years of experience
        num_questions: Number of AI questions to generate
    
    Returns:
        List of question dictionaries
    """
    if not _model_loaded:
        _load_model()
    
    if not _model_loaded or _model is None:
        logger.warning("SmolLM2 not available, returning fallback behavioral questions")
        return _generate_fallback_behavioral_questions(job_title, num_questions)
    
    try:
        import torch
        
        # Improved prompt with clear instructions and examples
        skills_context = f"Missing skills: {', '.join(skill_gaps[:2])}" if skill_gaps else ""
        strengths_context = f"Strong in: {', '.join(matched_skills[:2])}" if matched_skills else ""
        
        prompt = f"""Generate {num_questions} interview questions for a {job_title} candidate with {experience_years} years of experience.

{strengths_context}
{skills_context}

Requirements:
- Each question must be specific and actionable
- Focus on real-world scenarios and problem-solving
- Ask about collaboration, technical decisions, or past challenges
- Questions should be open-ended (not yes/no)

Example good questions:
- "Tell us about a time when you had to make a critical technical decision under pressure. What was your process?"
- "Describe a situation where you disagreed with a team member about an approach. How did you resolve it?"

Generate {num_questions} questions now, numbered 1-{num_questions}:"""

        device = _model.device
        inputs = _tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            outputs = _model.generate(
                inputs,
                max_new_tokens=400,
                temperature=0.8,
                top_p=0.92,
                do_sample=True,
                pad_token_id=_tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Get only the generated part (skip the prompt)
        generated_text = _tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        logger.info(f"AI generated text: {generated_text[:200]}...")
        
        # Parse questions from generated text
        questions = _parse_ai_questions(generated_text, num_questions, job_title)
        
        if len(questions) < num_questions:
            logger.warning(f"Only extracted {len(questions)}/{num_questions} AI questions, adding fallbacks")
            fallbacks = _generate_fallback_behavioral_questions(job_title, num_questions - len(questions))
            questions.extend(fallbacks)
        
        return questions
        
    except Exception as e:
        logger.error(f"❌ AI question generation failed: {e}")
        return _generate_fallback_behavioral_questions(job_title, num_questions)


def _generate_fallback_behavioral_questions(job_title: str, count: int) -> List[Dict[str, Any]]:
    """Generate fallback behavioral questions when AI fails"""
    fallback_pool = [
        {
            "text": f"Describe a complex technical problem you solved in a {job_title} role. What made it challenging and how did you approach it?",
            "category": "problem_solving"
        },
        {
            "text": "Tell us about a time when you had to learn a new technology quickly for a project. What was your learning strategy?",
            "category": "learning_agility"
        },
        {
            "text": "Describe a situation where you had to balance technical excellence with business deadlines. How did you make trade-offs?",
            "category": "decision_making"
        },
        {
            "text": "Share an example of when you received critical feedback on your work. How did you respond and what did you change?",
            "category": "growth_mindset"
        },
        {
            "text": "Tell us about a project where you collaborated with non-technical stakeholders. How did you communicate technical concepts?",
            "category": "communication"
        },
        {
            "text": f"What's the most innovative solution you've implemented in your work as a {job_title}? What impact did it have?",
            "category": "innovation"
        }
    ]
    
    questions = []
    for i in range(min(count, len(fallback_pool))):
        questions.append({
            "question_id": f"Q{100 + i}",
            "type": "text",
            "text": fallback_pool[i]["text"],
            "required": True,
            "category": fallback_pool[i]["category"],
            "generated_by": "Fallback Template",
            "placeholder": "Share specific details: the situation, your actions, and the measurable outcomes..."
        })
    
    return questions


def _parse_ai_questions(text: str, expected_count: int, job_title: str) -> List[Dict[str, Any]]:
    """Parse generated questions from AI output - improved version"""
    questions = []
    
    # Clean up the text
    text = text.strip()
    
    # Split by newlines and filter
    lines = text.split('\n')
    
    # Multiple patterns to catch questions
    question_patterns = [
        r'^(\d+)[\.\)]\s*(.+)',           # "1. Question" or "1) Question"
        r'^Q(\d+)[:\.\)]\s*(.+)',          # "Q1: Question" or "Q1. Question"
        r'^Question\s+(\d+)[:\.\)]\s*(.+)', # "Question 1: ..."
        r'^\-\s*(.+)',                     # "- Question"
    ]
    
    question_num = 100  # Start AI questions at Q100
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines or very short lines
        if not line or len(line) < 25:
            continue
        
        # Skip lines that look like instructions (common failure mode)
        if any(skip in line.lower() for skip in ['be specific', 'assess', 'should:', 'requirements:', 'format:', 'example']):
            continue
        
        question_text = None
        
        # Try each pattern
        for pattern in question_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                # Extract question text (last group)
                question_text = match.groups()[-1].strip()
                break
        
        # If no pattern matched but line looks like a question (ends with ?)
        if not question_text and line.endswith('?') and len(line) > 40:
            question_text = line
        
        # Add valid questions
        if question_text and len(question_text) > 30:
            # Clean up question text
            question_text = question_text.strip('"').strip()
            
            questions.append({
                "question_id": f"Q{question_num}",
                "type": "text",
                "text": question_text,
                "required": True,
                "category": "behavioral",
                "generated_by": "SmolLM2-1.7B-Instruct",
                "placeholder": "Share specific details: situation, actions taken, and outcomes achieved..."
            })
            question_num += 1
            
            if len(questions) >= expected_count:
                break
    
    logger.info(f"Extracted {len(questions)}/{expected_count} AI questions")
    return questions[:expected_count]


def generate_questionnaire(
    job_title: str,
    job_id: str,
    resume_id: str,
    candidate_email: str,
    candidate_name: str,
    match_data: Dict[str, Any],
    use_ai: bool = True
) -> Dict[str, Any]:
    """
    Generate complete questionnaire with hybrid approach
    
    Args:
        job_title: Job title
        job_id: Job ID
        resume_id: Resume ID
        candidate_email: Candidate email
        candidate_name: Candidate name
        match_data: Match results with skill_gaps, matched_skills, etc.
        use_ai: Whether to use AI for behavioral questions
    
    Returns:
        Complete questionnaire dictionary
    """
    skill_gaps = match_data.get('skill_gaps', [])
    matched_skills = match_data.get('matched_skills', [])
    experience_years = match_data.get('candidate_experience_years') or 0
    required_experience = match_data.get('required_experience_years') or 0
    
    # Ensure integers
    experience_years = int(experience_years) if experience_years else 0
    required_experience = int(required_experience) if required_experience else 0
    
    # Generate template questions (60% - focused and contextual)
    template_questions = generate_template_questions(
        job_title=job_title,
        skill_gaps=skill_gaps,
        matched_skills=matched_skills,
        experience_years=experience_years,
        required_experience=required_experience
    )
    
    # Generate AI questions (40% - behavioral and scenario-based)
    ai_questions = []
    if use_ai:
        # Aim for 3-4 AI questions to balance the questionnaire
        num_ai = max(3, len(template_questions) // 2)
        ai_questions = generate_ai_questions(
            job_title=job_title,
            skill_gaps=skill_gaps,
            matched_skills=matched_skills,
            experience_years=experience_years,
            num_questions=num_ai
        )
    
    # Combine and reorder question IDs sequentially
    all_questions = template_questions + ai_questions
    
    # Renumber questions sequentially
    for i, q in enumerate(all_questions, 1):
        q['question_id'] = f"Q{i}"
    
    # Generate questionnaire ID
    questionnaire_id = hashlib.md5(
        f"{job_id}{resume_id}{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:16]
    
    logger.info(f"Generated questionnaire: {len(template_questions)} template + {len(ai_questions)} AI = {len(all_questions)} total")
    
    return {
        "questionnaire_id": questionnaire_id,
        "job_id": job_id,
        "job_title": job_title,
        "candidate_email": candidate_email,
        "candidate_name": candidate_name,
        "resume_id": resume_id,
        "title": f"Pre-Interview Questionnaire: {job_title}",
        "description": f"Please answer the following questions to help us better understand your qualifications for the {job_title} position.",
        "questions": all_questions,
        "total_questions": len(all_questions),
        "template_questions": len(template_questions),
        "ai_questions": len(ai_questions),
        "created_at": datetime.utcnow(),
        "status": "active"
    }


# Singleton pattern for model
def get_question_generator():
    """Get question generator instance (lazy loads model)"""
    if not _model_loaded:
        _load_model()
    return {
        "generate_questionnaire": generate_questionnaire,
        "generate_template_questions": generate_template_questions,
        "generate_ai_questions": generate_ai_questions,
        "model_loaded": _model_loaded
    }
