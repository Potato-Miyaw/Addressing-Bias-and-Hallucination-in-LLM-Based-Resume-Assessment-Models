"""
Feedback Router - API endpoints for NER corrections
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feedback_handler import get_feedback_improver

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

# Pydantic models
class NERFeedback(BaseModel):
    resume_id: str
    field: str  # 'name', 'email', or 'phone'
    extracted: str  # What the NER extracted
    correct: str  # What it should have been
    resume_text: str  # Full resume text for pattern learning

class FeedbackResponse(BaseModel):
    success: bool
    field: str
    correction_count: int
    learned_pattern: Optional[str]
    total_patterns: int
    message: str

@router.post("/ner-correction")
async def submit_ner_correction(feedback: NERFeedback) -> FeedbackResponse:
    """
    Submit correction for NER extraction
    System learns patterns from corrections automatically
    """
    try:
        improver = get_feedback_improver()
        
        # Validate field
        if feedback.field not in ['name', 'email', 'phone']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid field: {feedback.field}. Must be 'name', 'email', or 'phone'"
            )
        
        # Add correction and learn pattern
        result = improver.add_correction(
            field=feedback.field,
            extracted=feedback.extracted,
            correct=feedback.correct,
            resume_text=feedback.resume_text,
            resume_id=feedback.resume_id
        )
        
        return FeedbackResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(e)}"
        )

@router.get("/stats")
async def get_feedback_stats():
    """Get feedback statistics and learned patterns"""
    try:
        improver = get_feedback_improver()
        stats = improver.get_stats()
        patterns = improver.export_patterns()
        
        return {
            "success": True,
            "stats": stats,
            "learned_patterns": patterns
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

@router.get("/patterns")
async def get_learned_patterns():
    """Get all learned patterns for each field"""
    try:
        improver = get_feedback_improver()
        patterns = improver.export_patterns()
        
        return {
            "success": True,
            "patterns": patterns,
            "pattern_counts": {
                field: len(patterns_list)
                for field, patterns_list in patterns.items()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get patterns: {str(e)}"
        )

@router.post("/test-pattern")
async def test_learned_patterns(field: str, text: str):
    """Test learned patterns on sample text"""
    try:
        if field not in ['name', 'email', 'phone']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid field: {field}"
            )
        
        improver = get_feedback_improver()
        matches = improver.apply_patterns(field, text)
        
        return {
            "success": True,
            "field": field,
            "matches": matches,
            "match_count": len(matches),
            "patterns_used": len(improver.learned_patterns[field])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test patterns: {str(e)}"
        )
