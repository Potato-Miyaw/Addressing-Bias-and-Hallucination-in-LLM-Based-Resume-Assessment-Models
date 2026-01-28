from deep_translator import GoogleTranslator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='en')
        
    def translate_to_english(self, text: str) -> dict:
        """Translate text to English and detect source language"""
        try:
            # First detect language (GoogleTranslator handles this automatically with source='auto')
            # But deep-translator doesn't expose confidence or detected lang directly in a simple way 
            # without making separate calls or using different logic.
            # For MVP, we will assume if it translates, it worked.
            
            # Limit text length to avoid timeouts/limits if necessary, but library handles chunks usually
            # optimizing: take first 500 chars to detect language if needed using another lib,
            # but for now rely on translation.
            
            translated_text = self.translator.translate(text)
            
            return {
                "original_text": text,
                "translated_text": translated_text,
                "success": True
            }
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                "original_text": text,
                "translated_text": text, # Fallback
                "success": False,
                "error": str(e)
            }

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate verification results back to original language"""
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            logger.error(f"Back-translation failed: {str(e)}")
            return text 
