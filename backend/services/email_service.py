"""
Email Service - Send questionnaire invitations to candidates
Supports SMTP (Gmail, Outlook, custom) with environment variable configuration
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending questionnaire invitations"""
    
    def __init__(self):
        """Initialize with environment variables"""
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.from_name = os.getenv("FROM_NAME", "HR Team")
        
        # Check if email is configured
        self.is_configured = bool(self.smtp_user and self.smtp_password)
        
        if not self.is_configured:
            logger.warning("⚠️ Email service not configured. Set SMTP_USER and SMTP_PASSWORD environment variables.")
    
    def send_questionnaire_invitation(
        self,
        candidate_email: str,
        candidate_name: str,
        invitation_link: str,
        job_title: str,
        expires_at: str,
        company_name: str = "Our Company"
    ) -> bool:
        """
        Send questionnaire invitation email to candidate
        
        Args:
            candidate_email: Candidate's email address
            candidate_name: Candidate's name
            invitation_link: Full questionnaire URL with token
            job_title: Job position title
            expires_at: Expiration datetime string
            company_name: Company name for branding
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_configured:
            logger.error("❌ Cannot send email: SMTP not configured")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"📋 Questionnaire for {job_title} Position"
            msg["From"] = f"{self.from_name} <{self.from_email}>"
            msg["To"] = candidate_email
            
            # Plain text version
            text_body = f"""
Hi {candidate_name},

Thank you for your interest in the {job_title} position at {company_name}.

As part of our hiring process, we would like you to complete a brief questionnaire to help us better understand your qualifications and experience.

Please click the link below to access the questionnaire:
{invitation_link}

This link will expire on {expires_at}.

If you have any questions, please don't hesitate to reach out.

Best regards,
{self.from_name}
{company_name}
"""
            
            # HTML version
            html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px 10px 0 0;
        }}
        .content {{
            background: #f9f9f9;
            padding: 30px;
            border-radius: 0 0 10px 10px;
        }}
        .button {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 5px;
            margin: 20px 0;
            font-weight: bold;
        }}
        .button:hover {{
            background: #5568d3;
        }}
        .info-box {{
            background: #fff;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #777;
            font-size: 12px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📋 Questionnaire Invitation</h1>
    </div>
    <div class="content">
        <p>Hi <strong>{candidate_name}</strong>,</p>
        
        <p>Thank you for your interest in the <strong>{job_title}</strong> position at <strong>{company_name}</strong>.</p>
        
        <p>As part of our hiring process, we would like you to complete a brief questionnaire to help us better understand your qualifications and experience.</p>
        
        <div style="text-align: center;">
            <a href="{invitation_link}" class="button">📝 Complete Questionnaire</a>
        </div>
        
        <div class="info-box">
            <p><strong>⏰ Important:</strong> This link will expire on <strong>{expires_at}</strong></p>
            <p><strong>⚡ Estimated time:</strong> 10-15 minutes</p>
        </div>
        
        <p>If the button doesn't work, copy and paste this link into your browser:</p>
        <p style="word-break: break-all; background: #fff; padding: 10px; border-radius: 5px;">
            {invitation_link}
        </p>
        
        <p>If you have any questions, please don't hesitate to reach out.</p>
        
        <p>Best regards,<br>
        <strong>{self.from_name}</strong><br>
        {company_name}</p>
    </div>
    <div class="footer">
        <p>This is an automated message. Please do not reply to this email.</p>
    </div>
</body>
</html>
"""
            
            # Attach both versions
            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()  # Secure connection
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"✅ Questionnaire invitation sent to {candidate_email}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error("❌ SMTP Authentication failed. Check your credentials.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"❌ SMTP error: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False
    
    def send_test_email(self, test_recipient: str) -> bool:
        """
        Send a test email to verify configuration
        
        Args:
            test_recipient: Email address to send test to
        
        Returns:
            True if sent successfully
        """
        return self.send_questionnaire_invitation(
            candidate_email=test_recipient,
            candidate_name="Test User",
            invitation_link="http://localhost:8501/Questionnaire_Response?token=test123",
            job_title="Test Position",
            expires_at="2026-02-01",
            company_name="Test Company"
        )


# Singleton instance
_email_service = None


def get_email_service() -> EmailService:
    """Get or create email service singleton"""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
