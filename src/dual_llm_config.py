import os
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from typing import Tuple
import yaml


class DualLLMConfigurator:
    def __init__(self, config_path: str = "./config/thermodynamics_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n사용 디바이스: {self.device}\n")
    
    def setup_embedding_models(self) -> Tuple[HuggingFaceEmbedding, HuggingFaceEmbedding]:
        embed_configs = self.config['embedding_models']
        
        # 한국어 임베딩 모델
        print(f"\n한국어 임베딩 모델 로딩: {embed_configs['korean']['name']}")
        korean_embed = HuggingFaceEmbedding(
            model_name=embed_configs['korean']['name'],
            device=self.device,
            max_length=embed_configs['korean'].get('max_length', 512),
            trust_remote_code=True,
            model_kwargs={'default_task': 'retrieval'}
        )
        print("한국어 임베딩 모델 로드 완료\n")
        
        # 영어 임베딩 모델
        print(f"\n영어/다국어 임베딩 모델 로딩: {embed_configs['english']['name']}")
        english_embed = HuggingFaceEmbedding(
            model_name=embed_configs['english']['name'],
            device=self.device,
            max_length=embed_configs['english'].get('max_length', 512),
            trust_remote_code=True,
            model_kwargs={'default_task': 'retrieval'}
        )
        print("영어 임베딩 모델 로드 완료\n")
        
        return korean_embed, english_embed
    
    def setup_llm(self) -> Gemini:
        llm_config = self.config['llm_model']
        
        print(f"\nLLM 모델 로딩: {llm_config['name']}")
        llm = Gemini(
            model=llm_config["name"],
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=llm_config.get("temperature", 0.7)
        )
        
        print("LLM 모델 로드 완료\n")
        return llm