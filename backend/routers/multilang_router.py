from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.translation_service import TranslationService
from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.utils.document_parser import extract_text_from_file

router = APIRouter(prefix="/api/multilang", tags=["Multi-Language"])

translation_service = TranslationService()
ner_extractor = ResumeNERExtractor()

@router.post("/upload")
async def upload_foreign_resume(
    file: UploadFile = File(...), 
    target_lang: str = Form("en") # Default parse target is English
):
    try:
        content = await file.read()
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text:
            raise HTTPException(400, "Could not extract text")

        # 1. Translate to English
        trans_result = translation_service.translate_to_english(resume_text)
        english_text = trans_result["translated_text"]
        
        # 2. Extract Info (using English text)
        resume_data = ner_extractor.parse_resume(english_text)
        
        # 3. Translate specific fields back to original language for display? 
        # Or just show English extraction? User requirement: "displays results in original language"
        # We need to detect source language properly to translate back.
        # deep-translator's 'auto' works for src, but we assume we want to support ES, FR, DE.
        # Ideally we let user pick language or just translate results to their UI language.
        # For now, let's assume we translate SUMMARY/MESSAGE back. 
        # Fields like Name, Email are universal. 
        # Skills/Job Titles might be better in English for matching, but display in original?
        # That's complex. Let's start by translating the "Message" and "Status".
        
        # NOTE: Since we don't strictly know source lang code from deep-translator immediately without extra call,
        # we might rely on client side or just default to English display if auto failing.
        # But let's try to detect if possible or just pass-through.
        
        return {
            "original_text_preview": resume_text[:200],
            "english_text_preview": english_text[:200],
            "parsed_data": resume_data,
            "translation_status": trans_result["success"]
        }

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.post("/process-text")
async def process_resume_text(
    payload: dict  # Expecting {"text": "..."}
):
    try:
        resume_text = payload.get("text")
        
        if not resume_text:
            raise HTTPException(400, "No text provided")

        # 1. Translate to English
        trans_result = translation_service.translate_to_english(resume_text)
        english_text = trans_result["translated_text"]
        
        # 2. Extract Info (using English text)
        resume_data = ner_extractor.parse_resume(english_text)
        
        return {
            "original_text_preview": resume_text[:200],
            "english_text_preview": english_text[:200],
            "english_full_text": english_text,
            "parsed_data": resume_data,
            "translation_status": trans_result["success"]
        }

    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
