import os
from typing import Optional
import google.generativeai as genai


class QueryTranslator:
    def __init__(self, config: dict):
        self.translation_config = config.get('query_translation', {})
        self.enabled = self.translation_config.get('enabled', True)
        self.translate_for_textbook = self.translation_config.get('translate_for_textbook', True)
        
        if self.enabled:
            # Gemini API 설정
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.translation_model = genai.GenerativeModel('gemini-2.0-flash')
            print("\n쿼리 번역기 초기화 완료\n")
    
    def translate_to_english(self, korean_query: str) -> str:
        if not self.enabled or not self.translate_for_textbook:
            return korean_query
        
        prompt = f"""Translate the following Korean query about thermodynamics into English.
Keep technical terms accurate and maintain the meaning.
Only output the English translation, nothing else.

Korean Query: {korean_query}

English Translation:"""
        
        try:
            response = self.translation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # 번역은 낮은 temperature
                    max_output_tokens=256,
                )
            )
            
            translated = response.text.strip()
            print(f"번역: '{korean_query}' → '{translated}'")
            return translated
            
        except Exception as e:
            print(f"번역 실패, 원본 사용: {e}")
            return korean_query
    
    def is_enabled(self) -> bool:
        return self.enabled and self.translate_for_textbook