"""
Questionnaire Router - API endpoints for questionnaire management
Handles creation, invitation, token validation, and response submission
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import secrets
import logging

from backend.database import (
    save_questionnaire,
    get_questionnaire,
    list_questionnaires,
    delete_questionnaire,
    save_invitation,
    get_invitation_by_token,
    mark_invitation_used,
    get_invitations_by_questionnaire,
    save_response,
    get_response_by_token,
    get_responses_by_questionnaire,
    get_match
)
from backend.services.question_generator import generate_questionnaire
from backend.services.email_service import get_email_service
from backend.services.whatsapp_service import get_whatsapp_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/questionnaire", tags=["Questionnaire"])


# ==================== Pydantic Models ====================

class QuestionCreate(BaseModel):
    """Question model"""
    question_id: str
    type: str = Field(pattern="^(text|multiple_choice|rating|number)$")
    text: str
    options: Optional[List[str]] = None
    required: bool = True
    category: Optional[str] = None


class QuestionnaireGenerate(BaseModel):
    """Request to auto-generate questionnaire from match"""
    match_id: str
    use_ai: bool = True  # Use SmolLM2 for behavioral questions


class QuestionnaireCreate(BaseModel):
    """Manual questionnaire creation"""
    title: str
    description: str
    questions: List[QuestionCreate]
    job_id: Optional[str] = None
    status: str = Field(default="active", pattern="^(active|archived)$")


class InvitationCreate(BaseModel):
    """Send questionnaire invitation"""
    questionnaire_id: str
    candidate_email: EmailStr
    candidate_name: str
    candidate_phone: Optional[str] = None  # For WhatsApp (E.164 format: +1234567890)
    delivery_method: str = Field(default="email", pattern="^(email|whatsapp|both)$")
    job_id: Optional[str] = None
    resume_id: Optional[str] = None
    expires_in_days: int = Field(default=7, ge=1, le=30)


class ResponseSubmit(BaseModel):
    """Submit questionnaire response"""
    token: str
    answers: List[Dict[str, Any]]  # [{question_id: str, answer: any}]


class ResendEmailRequest(BaseModel):
    """Resend email invitation"""
    token: str
    candidate_email: EmailStr
    candidate_name: str
    invitation_link: str
    job_title: str
    expires_at: str


# ==================== Endpoints ====================

@router.post("/generate")
async def generate_questionnaire_from_match(request: QuestionnaireGenerate):
    """
    Auto-generate questionnaire from match results using SmolLM2
    
    - **match_id**: Match ID to generate questions from
    - **use_ai**: Use SmolLM2 for behavioral questions (default: True)
    """
    try:
        # Get match data
        match = await get_match(request.match_id)
        if not match:
            raise HTTPException(404, f"Match not found: {request.match_id}")
        
        # Generate questionnaire
        questionnaire_data = generate_questionnaire(
            job_title=match.get('job_title', 'Position'),
            job_id=match['job_id'],
            resume_id=match['resume_id'],
            candidate_email=match['candidate_email'],
            candidate_name=match['candidate_name'],
            match_data={
                'skill_gaps': match['missing_skills'],
                'matched_skills': match['matched_skills'],
                'candidate_experience_years': match.get('candidate_experience_years', 0),
                'required_experience_years': match.get('required_experience_years', 0)
            },
            use_ai=request.use_ai
        )
        
        # Save to database
        success = await save_questionnaire(questionnaire_data)
        if not success:
            raise HTTPException(500, "Failed to save questionnaire")
        
        logger.info(f"✅ Generated questionnaire: {questionnaire_data['questionnaire_id']}")
        
        return {
            "success": True,
            "questionnaire": questionnaire_data,
            "message": f"Generated {questionnaire_data['total_questions']} questions ({questionnaire_data['template_questions']} template + {questionnaire_data['ai_questions']} AI)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating questionnaire: {e}")
        raise HTTPException(500, str(e))


@router.post("/create")
async def create_questionnaire(request: QuestionnaireCreate):
    """
    Manually create a custom questionnaire
    
    - **title**: Questionnaire title
    - **description**: Description for candidates
    - **questions**: List of questions
    """
    try:
        # Generate ID
        questionnaire_id = hashlib.md5(
            f"{request.title}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        questionnaire_data = {
            "questionnaire_id": questionnaire_id,
            "title": request.title,
            "description": request.description,
            "questions": [q.model_dump() for q in request.questions],
            "total_questions": len(request.questions),
            "job_id": request.job_id,
            "status": request.status,
            "created_at": datetime.utcnow()
        }
        
        success = await save_questionnaire(questionnaire_data)
        if not success:
            raise HTTPException(500, "Failed to save questionnaire")
        
        return {
            "success": True,
            "questionnaire_id": questionnaire_id,
            "questionnaire": questionnaire_data
        }
        
    except Exception as e:
        logger.error(f"Error creating questionnaire: {e}")
        raise HTTPException(500, str(e))


@router.get("/{questionnaire_id}")
async def get_questionnaire_by_id(questionnaire_id: str):
    """Get questionnaire by ID"""
    questionnaire = await get_questionnaire(questionnaire_id)
    if not questionnaire:
        raise HTTPException(404, f"Questionnaire not found: {questionnaire_id}")
    
    return {
        "success": True,
        "questionnaire": questionnaire
    }


@router.get("/")
async def list_all_questionnaires(
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None, pattern="^(active|archived)$")
):
    """List questionnaires with optional status filter"""
    questionnaires = await list_questionnaires(limit=limit, skip=skip, status=status)
    
    return {
        "success": True,
        "count": len(questionnaires),
        "limit": limit,
        "skip": skip,
        "questionnaires": questionnaires
    }


@router.delete("/{questionnaire_id}")
async def delete_questionnaire_by_id(questionnaire_id: str):
    """Delete a questionnaire"""
    success = await delete_questionnaire(questionnaire_id)
    if not success:
        raise HTTPException(404, f"Questionnaire not found: {questionnaire_id}")
    
    return {
        "success": True,
        "message": f"Questionnaire {questionnaire_id} deleted"
    }


@router.post("/invite")
async def send_invitation(request: InvitationCreate):
    """
    Send questionnaire invitation to candidate
    Generates unique token and creates invitation link
    
    - **questionnaire_id**: Questionnaire to send
    - **candidate_email**: Candidate's email
    - **candidate_name**: Candidate's name
    - **expires_in_days**: Token expiry (1-30 days)
    """
    try:
        # Verify questionnaire exists
        questionnaire = await get_questionnaire(request.questionnaire_id)
        if not questionnaire:
            raise HTTPException(404, f"Questionnaire not found: {request.questionnaire_id}")
        
        # Generate unique token
        token = secrets.token_urlsafe(32)
        
        # Generate invitation ID
        invitation_id = hashlib.md5(
            f"{request.questionnaire_id}{request.candidate_email}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
        
        # Create invitation
        invitation_data = {
            "invitation_id": invitation_id,
            "token": token,
            "questionnaire_id": request.questionnaire_id,
            "candidate_email": request.candidate_email,
            "candidate_name": request.candidate_name,
            "job_id": request.job_id,
            "resume_id": request.resume_id,
            "expires_at": expires_at,
            "used": False,
            "used_at": None,
            "sent_at": datetime.utcnow(),
            "created_by_hr": "hr_user"  # TODO: Get from auth
        }
        
        success = await save_invitation(invitation_data)
        if not success:
            raise HTTPException(500, "Failed to save invitation")
        
        # Generate invitation link
        # TODO: In production, use actual domain
        invitation_link = f"http://localhost:8501/Questionnaire_Response?token={token}"
        
        # Send via selected method(s)
        email_sent = False
        whatsapp_sent = False
        delivery_method = request.delivery_method
        
        # Send Email
        if delivery_method in ["email", "both"]:
            email_service = get_email_service()
            if email_service.is_configured:
                try:
                    email_sent = email_service.send_questionnaire_invitation(
                        candidate_email=request.candidate_email,
                        candidate_name=request.candidate_name,
                        invitation_link=invitation_link,
                        job_title=questionnaire.get('title', 'Position'),
                        expires_at=expires_at.strftime('%Y-%m-%d %H:%M UTC'),
                        company_name="Your Company"  # TODO: Make configurable
                    )
                    
                    if email_sent:
                        logger.info(f"✅ Email sent to {request.candidate_email}")
                    else:
                        logger.warning(f"⚠️ Failed to send email to {request.candidate_email}")
                except Exception as e:
                    logger.error(f"❌ Email sending error: {e}")
            else:
                logger.warning("⚠️ Email not configured")
        
        # Send WhatsApp
        if delivery_method in ["whatsapp", "both"]:
            if not request.candidate_phone:
                logger.warning("⚠️ WhatsApp requested but no phone number provided")
            else:
                whatsapp_service = get_whatsapp_service()
                if whatsapp_service.is_configured:
                    try:
                        whatsapp_sent = whatsapp_service.send_questionnaire_invitation(
                            candidate_phone=request.candidate_phone,
                            candidate_name=request.candidate_name,
                            invitation_link=invitation_link,
                            job_title=questionnaire.get('title', 'Position'),
                            expires_at=expires_at.strftime('%Y-%m-%d %H:%M UTC'),
                            company_name="Your Company"  # TODO: Make configurable
                        )
                        
                        if whatsapp_sent:
                            logger.info(f"✅ WhatsApp sent to {request.candidate_phone}")
                        else:
                            logger.warning(f"⚠️ Failed to send WhatsApp to {request.candidate_phone}")
                    except Exception as e:
                        logger.error(f"❌ WhatsApp sending error: {e}")
                else:
                    logger.warning("⚠️ WhatsApp not configured")
        
        logger.info(f"✅ Created invitation for {request.candidate_email}")
        
        return {
            "success": True,
            "invitation_id": invitation_id,
            "token": token,
            "invitation_link": invitation_link,
            "expires_at": expires_at,
            "email_sent": email_sent,
            "whatsapp_sent": whatsapp_sent,
            "message": f"Invitation {'sent' if (email_sent or whatsapp_sent) else 'created'} for {request.candidate_email}. Link expires in {request.expires_in_days} days."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating invitation: {e}")
        raise HTTPException(500, str(e))


@router.get("/validate-token/{token}")
async def validate_token(token: str):
    """
    Validate invitation token and return questionnaire
    Used by candidates when accessing the link
    
    - **token**: Invitation token from URL
    """
    try:
        # Get invitation
        invitation = await get_invitation_by_token(token)
        if not invitation:
            raise HTTPException(404, "Invalid or expired invitation link")
        
        # Check if already used
        if invitation['used']:
            # Check if response exists
            response = await get_response_by_token(token)
            if response:
                return {
                    "success": False,
                    "error": "already_submitted",
                    "message": "You have already submitted this questionnaire",
                    "submitted_at": response['submitted_at']
                }
        
        # Check expiry
        if datetime.utcnow() > invitation['expires_at']:
            return {
                "success": False,
                "error": "expired",
                "message": "This invitation link has expired",
                "expired_at": invitation['expires_at']
            }
        
        # Get questionnaire
        questionnaire = await get_questionnaire(invitation['questionnaire_id'])
        if not questionnaire:
            raise HTTPException(404, "Questionnaire not found")
        
        return {
            "success": True,
            "valid": True,
            "invitation": {
                "candidate_name": invitation['candidate_name'],
                "candidate_email": invitation['candidate_email'],
                "job_id": invitation.get('job_id'),
                "expires_at": invitation['expires_at']
            },
            "questionnaire": questionnaire
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating token: {e}")
        raise HTTPException(500, str(e))


@router.post("/submit")
async def submit_response(request: ResponseSubmit):
    """
    Submit questionnaire response
    
    - **token**: Invitation token
    - **answers**: List of {question_id, answer} pairs
    """
    try:
        # Validate token
        invitation = await get_invitation_by_token(request.token)
        if not invitation:
            raise HTTPException(404, "Invalid invitation")
        
        # Check if already submitted
        existing_response = await get_response_by_token(request.token)
        if existing_response:
            raise HTTPException(400, "Response already submitted")
        
        # Check expiry
        if datetime.utcnow() > invitation['expires_at']:
            raise HTTPException(400, "Invitation has expired")
        
        # Generate response ID
        response_id = hashlib.md5(
            f"{request.token}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Create response
        response_data = {
            "response_id": response_id,
            "invitation_id": invitation['invitation_id'],
            "token": request.token,
            "questionnaire_id": invitation['questionnaire_id'],
            "candidate_email": invitation['candidate_email'],
            "candidate_name": invitation['candidate_name'],
            "answers": request.answers,
            "submitted_at": datetime.utcnow()
        }
        
        # Save response
        success = await save_response(response_data)
        if not success:
            raise HTTPException(500, "Failed to save response")
        
        # Mark invitation as used
        await mark_invitation_used(request.token)
        
        logger.info(f"✅ Response submitted by {invitation['candidate_email']}")
        
        return {
            "success": True,
            "response_id": response_id,
            "message": "Thank you! Your response has been submitted successfully."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting response: {e}")
        raise HTTPException(500, str(e))


@router.get("/invitations/{questionnaire_id}")
async def get_invitations(questionnaire_id: str):
    """Get all invitations for a questionnaire"""
    invitations = await get_invitations_by_questionnaire(questionnaire_id)
    
    return {
        "success": True,
        "count": len(invitations),
        "invitations": invitations
    }


@router.get("/responses/{questionnaire_id}")
async def get_responses(questionnaire_id: str):
    """Get all responses for a questionnaire"""
    responses = await get_responses_by_questionnaire(questionnaire_id)
    
    return {
        "success": True,
        "count": len(responses),
        "responses": responses
    }


@router.get("/email/status")
async def get_email_status():
    """Check email service configuration status"""
    email_service = get_email_service()
    
    return {
        "configured": email_service.is_configured,
        "smtp_host": email_service.smtp_host if email_service.is_configured else None,
        "smtp_port": email_service.smtp_port if email_service.is_configured else None,
        "from_email": email_service.from_email if email_service.is_configured else None,
        "message": "Email service is configured and ready" if email_service.is_configured else "Email service not configured. Set SMTP_USER and SMTP_PASSWORD in .env file."
    }


@router.post("/email/test")
async def test_email(test_email: EmailStr):
    """
    Send a test email to verify configuration
    
    - **test_email**: Email address to send test to
    """
    email_service = get_email_service()
    
    if not email_service.is_configured:
        raise HTTPException(400, "Email service not configured. Set SMTP credentials in .env file.")
    
    try:
        success = email_service.send_test_email(test_email)
        
        if success:
            return {
                "success": True,
                "message": f"Test email sent successfully to {test_email}"
            }
        else:
            raise HTTPException(500, "Failed to send test email. Check backend logs for details.")
    
    except Exception as e:
        logger.error(f"Test email failed: {e}")
        raise HTTPException(500, f"Email test failed: {str(e)}")


@router.post("/resend-email")
async def resend_email(request: ResendEmailRequest):
    """
    Resend email invitation (useful if initial send failed)
    
    - **token**: Invitation token
    - **candidate_email**: Candidate's email
    - **candidate_name**: Candidate's name
    - **invitation_link**: Full invitation URL
    - **job_title**: Job title for email
    - **expires_at**: Expiration datetime string
    """
    email_service = get_email_service()
    
    if not email_service.is_configured:
        raise HTTPException(400, "Email service not configured. Set SMTP credentials in .env file.")
    
    try:
        # Verify invitation exists and is valid
        invitation = await get_invitation_by_token(request.token)
        if not invitation:
            raise HTTPException(404, "Invalid invitation token")
        
        # Try to send email
        success = email_service.send_questionnaire_invitation(
            candidate_email=request.candidate_email,
            candidate_name=request.candidate_name,
            invitation_link=request.invitation_link,
            job_title=request.job_title,
            expires_at=request.expires_at,
            company_name="Your Company"  # TODO: Make configurable
        )
        
        if success:
            logger.info(f"✅ Resent email to {request.candidate_email}")
            return {
                "success": True,
                "message": f"Email resent successfully to {request.candidate_email}"
            }
        else:
            logger.error(f"❌ Failed to resend email to {request.candidate_email}")
            raise HTTPException(500, "Failed to send email. Check SMTP configuration and backend logs.")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resend email error: {e}")
        raise HTTPException(500, f"Failed to resend email: {str(e)}")


@router.get("/whatsapp/status")
async def whatsapp_status():
    """
    Check WhatsApp (Twilio) configuration status
    """
    whatsapp_service = get_whatsapp_service()
    
    return {
        "configured": whatsapp_service.is_configured,
        "message": "WhatsApp service is configured" if whatsapp_service.is_configured else "WhatsApp not configured. Set Twilio credentials in .env file. See WHATSAPP_SETUP.md for instructions.",
        "required_env_vars": [
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "TWILIO_WHATSAPP_NUMBER"
        ]
    }
