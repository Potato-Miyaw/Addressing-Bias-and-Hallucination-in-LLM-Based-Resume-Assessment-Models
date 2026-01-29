"""
WhatsApp Service - Send questionnaire invitations via WhatsApp using Twilio
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WhatsAppService:
    """WhatsApp service for sending questionnaire invitations via Twilio"""
    
    def __init__(self):
        """Initialize with environment variables"""
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
        
        # Check if WhatsApp is configured
        self.is_configured = bool(self.account_sid and self.auth_token)
        
        if self.is_configured:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("✅ WhatsApp service configured (Twilio)")
            except ImportError:
                logger.warning("⚠️ Twilio package not installed. Run: pip install twilio")
                self.is_configured = False
            except Exception as e:
                logger.error(f"❌ Failed to initialize Twilio client: {e}")
                self.is_configured = False
        else:
            logger.warning("⚠️ WhatsApp not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
            self.client = None
    
    def send_questionnaire_invitation(
        self,
        candidate_phone: str,
        candidate_name: str,
        invitation_link: str,
        job_title: str,
        expires_at: str,
        company_name: str = "Our Company"
    ) -> bool:
        """
        Send questionnaire invitation via WhatsApp
        
        Args:
            candidate_phone: Candidate's phone number (E.164 format: +1234567890)
            candidate_name: Candidate's name
            invitation_link: Full questionnaire URL with token
            job_title: Job position title
            expires_at: Expiration datetime string
            company_name: Company name for branding
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured:
            logger.error("❌ Cannot send WhatsApp: Twilio not configured")
            return False
        
        try:
            # Ensure phone number has whatsapp: prefix and proper format
            to_number = candidate_phone
            if not to_number.startswith("whatsapp:"):
                to_number = f"whatsapp:{to_number}"
            
            # Create message
            message_body = f"""Hi {candidate_name},

Thank you for your interest in the *{job_title}* position at {company_name}.

Please complete this brief questionnaire:
{invitation_link}

⏰ Link expires: {expires_at}
⚡ Takes 10-15 minutes

Reply if you have any questions!

Best regards,
{company_name} HR Team"""
            
            # Send message via Twilio
            message = self.client.messages.create(
                from_=self.from_number,
                body=message_body,
                to=to_number
            )
            
            logger.info(f"✅ WhatsApp sent to {candidate_phone} (SID: {message.sid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send WhatsApp to {candidate_phone}: {e}")
            return False
    
    def send_test_message(self, test_phone: str) -> bool:
        """
        Send a test WhatsApp message to verify configuration
        
        Args:
            test_phone: Phone number to send test to (E.164 format)
        
        Returns:
            True if sent successfully
        """
        return self.send_questionnaire_invitation(
            candidate_phone=test_phone,
            candidate_name="Test User",
            invitation_link="http://localhost:8501/Questionnaire_Response?token=test123",
            job_title="Test Position",
            expires_at="2026-02-01",
            company_name="Test Company"
        )


# Singleton instance
_whatsapp_service = None


def get_whatsapp_service() -> WhatsAppService:
    """Get or create WhatsApp service singleton"""
    global _whatsapp_service
    if _whatsapp_service is None:
        _whatsapp_service = WhatsAppService()
    return _whatsapp_service
