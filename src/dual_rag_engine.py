"""수정된 DualRAGEngine - 지식베이스 선택 및 쿼리 번역 지원"""

from typing import List, Dict, Optional
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
import yaml
from pathlib import Path
from src.query_translator import QueryTranslator


class DualRAGEngine:
    def __init__(
        self,
        lecture_index: Optional[VectorStoreIndex],
        textbook_index: Optional[VectorStoreIndex],
        lecture_nodes: Optional[List],
        textbook_nodes: Optional[List],
        llm: LLM,
        korean_embed: Optional[BaseEmbedding],
        english_embed: Optional[BaseEmbedding],
        config_path: str = "./config/thermodynamics_config.yaml"
    ):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 지식베이스 선택 설정
        kb_selection = self.config.get('knowledge_base_selection', {})
        self.use_lecture = kb_selection.get('use_lecture_notes', True)
        self.use_textbook = kb_selection.get('use_textbook', True)
        
        # 사용할 지식베이스에 따라 필수 객체 확인
        if self.use_lecture:
            if lecture_index is None or lecture_nodes is None or korean_embed is None:
                raise ValueError("강의자료 사용 설정이 활성화되었으나 필요한 객체가 없습니다.")
        
        if self.use_textbook:
            if textbook_index is None or textbook_nodes is None or english_embed is None:
                raise ValueError("교재 사용 설정이 활성화되었으나 필요한 객체가 없습니다.")
        
        self.lecture_index = lecture_index
        self.textbook_index = textbook_index
        self.lecture_nodes = lecture_nodes
        self.textbook_nodes = textbook_nodes
        self.llm = llm
        self.korean_embed = korean_embed
        self.english_embed = english_embed
        self.rag_config = self.config['rag']
        
        # 쿼리 번역기 초기화
        self.translator = QueryTranslator(self.config)
        
        # 검색 가중치
        self.weights = self.rag_config.get('retrieval_weights', {
            'lecture_notes': 0.6,
            'textbook': 0.4
        })
        
        # 사용 모드 출력
        self._print_mode()
        
        # 쿼리 엔진 초기화
        self.query_engine = self._create_query_engine()
    
    def _print_mode(self):
        print("\n" + "#" * 60)
        print("지식베이스 사용 모드")
        print("#" * 60)
        
        if not self.use_lecture and not self.use_textbook:
            print("LLM Only 모드 (지식베이스 사용 안함)")
        elif self.use_lecture and not self.use_textbook:
            print("강의자료만 사용")
        elif not self.use_lecture and self.use_textbook:
            print("교재만 사용")
            if self.translator.is_enabled():
                print("   + 쿼리 번역 활성화 (한국어 → 영어)")
        else:
            print("강의자료 + 교재 모두 사용")
            if self.translator.is_enabled():
                print("   + 쿼리 번역 활성화 (교재 검색용)")
        
        print("#" * 60 + "\n")
    
    def _create_fusion_retriever(
        self, 
        index: VectorStoreIndex, 
        nodes: List, 
        embed_model: BaseEmbedding,
        query: Optional[str] = None
    ) -> QueryFusionRetriever:
        top_k = self.rag_config.get('similarity_top_k', 5)
        
        # Dense Retriever
        dense = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
            embed_model=embed_model
        )
        
        # BM25 Retriever
        bm25 = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k
        )
        
        # Query Generation Prompt
        QUERY_GEN_PROMPT = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )
        
        # Fusion Retriever
        fusion = QueryFusionRetriever(
            retrievers=[dense, bm25],
            similarity_top_k=top_k,
            num_queries=3,
            mode='reciprocal_rerank',
            use_async=True,
            verbose=False,
            llm=self.llm,
            query_gen_prompt=QUERY_GEN_PROMPT,
        )
        
        return fusion
    
    def _create_query_engine(self) -> Optional[RetrieverQueryEngine]:
        # LLM Only 모드
        if not self.use_lecture and not self.use_textbook:
            print("LLM Only 모드 - Retriever 없이 LLM만 사용")
            return None
        
        # 강의자료만 사용
        if self.use_lecture and not self.use_textbook:
            print("강의자료 전용 쿼리 엔진 생성 중...")
            retriever = self._create_fusion_retriever(
                self.lecture_index, 
                self.lecture_nodes, 
                self.korean_embed
            )
            return self._build_query_engine(retriever, "lecture_only")
        
        # 교재만 사용
        if not self.use_lecture and self.use_textbook:
            print("교재 전용 쿼리 엔진 생성 중...")
            retriever = self._create_fusion_retriever(
                self.textbook_index, 
                self.textbook_nodes, 
                self.english_embed
            )
            return self._build_query_engine(retriever, "textbook_only")
        
        # 강의자료 + 교재 모두 사용
        print("통합 쿼리 엔진 생성 중...")
        
        # 강의자료 Fusion
        lecture_fusion = self._create_fusion_retriever(
            self.lecture_index, 
            self.lecture_nodes, 
            self.korean_embed
        )
        
        # 교재 Fusion
        textbook_fusion = self._create_fusion_retriever(
            self.textbook_index, 
            self.textbook_nodes, 
            self.english_embed
        )
        
        # 통합 Retriever
        top_k = self.rag_config.get('similarity_top_k', 5)
        unified_retriever = QueryFusionRetriever(
            retrievers=[lecture_fusion, textbook_fusion],
            similarity_top_k=top_k * 2,
            num_queries=1,
            mode='reciprocal_rerank',
            use_async=True,
            verbose=False,
        )
        
        return self._build_query_engine(unified_retriever, "dual")
    
    def _build_query_engine(self, retriever, mode: str) -> RetrieverQueryEngine:
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=self.rag_config.get('response_mode', 'compact'),
            use_async=False
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
        
        # 프롬프트 설정
        qa_prompt = self._get_qa_prompt_template(mode)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
        
        print(f"쿼리 엔진 생성 완료 (모드: {mode})")
        return query_engine
    
    def _get_qa_prompt_template(self, mode: str) -> PromptTemplate:
        prompt_path = self.config["prompt"]["qa_prompt"]
        
        # 모드별 기본 프롬프트
        if mode == "lecture_only":
            default_template = """당신은 열역학 과목을 가르치는 AI 조교입니다.
아래 제공된 강의자료를 바탕으로 학생의 질문에 답변해주세요.

강의자료 정보:
{context_str}

질문: {query_str}

답변 시 다음 사항을 준수하세요:
1. 강의자료의 내용을 바탕으로 정확하게 답변하세요
2. 수식이나 전문 용어는 정확하게 표기하세요
3. 명확하고 체계적으로 답변하세요
4. 주어진 문서로 답변이 불가능하면, 이를 고지하고 스스로 생각해서 답변하세요
5. 주어진 문서로 답변이 가능하다면, 출처를 표시하세요

답변:"""
        
        elif mode == "textbook_only":
            default_template = """당신은 열역학 과목을 가르치는 AI 조교입니다.
아래 제공된 교재 내용을 바탕으로 학생의 질문에 답변해주세요.

교재 정보 (영어 원문):
{context_str}

질문: {query_str}

답변 시 다음 사항을 준수하세요:
1. 영어 교재의 내용을 한국어로 번역하여 설명하세요
2. 수식이나 전문 용어는 정확하게 표기하세요
3. 명확하고 체계적으로 답변하세요
4. 주어진 문서로 답변이 불가능하면, 이를 고지하고 스스로 생각해서 답변하세요
5. 주어진 문서로 답변이 가능하다면, 출처를 표시하세요

답변:"""
        
        else:  # dual mode
            default_template = """당신은 열역학 과목을 가르치는 AI 조교입니다.
아래 제공된 문서 정보를 바탕으로 학생의 질문에 답변해주세요.

문서는 두 가지 출처에서 제공됩니다:
1. 강의자료 (한국어) - 교수님의 강의 노트 및 슬라이드
2. 교재 (영어) - 공식 교과서 내용

문서 정보:
{context_str}

질문: {query_str}

답변 시 다음 사항을 준수하세요:
1. 강의자료와 교재의 내용을 모두 참고하여 종합적으로 답변하세요
2. 영어 교재의 내용은 한국어로 번역하여 설명하세요
3. 수식이나 전문 용어는 정확하게 표기하세요
4. 필요시 강의자료와 교재의 설명을 비교하여 제공하세요
5. 주어진 문서로 답변이 불가능하면, 이를 고지하고 스스로 생각해서 답변하세요
6. 주어진 문서로 답변이 가능하다면, 출처를 표시하세요

답변:"""
        
        try:
            if Path(prompt_path).exists():
                template = Path(prompt_path).open().read()
            else:
                template = default_template
        except:
            template = default_template
        
        return PromptTemplate(template)
    
    def query(self, question: str) -> Dict:
        try:
            # LLM Only 모드
            if self.query_engine is None:
                return self._query_llm_only(question)
            
            # 교재 검색이 포함된 경우, 쿼리 번역
            translated_query = question
            if self.use_textbook and self.translator.is_enabled():
                translated_query = self.translator.translate_to_english(question)
            
            # RAG 쿼리 수행
            # 교재만 사용하는 경우 번역된 쿼리 사용
            if not self.use_lecture and self.use_textbook:
                response = self.query_engine.query(translated_query)
            else:
                # 강의자료 포함 시 원본 쿼리 사용
                response = self.query_engine.query(question)
            
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            sources = []
            for node in source_nodes:
                metadata = node.metadata
                kb_type = metadata.get('kb_type', 'unknown')
                language = metadata.get('language', 'unknown')
                
                # 기본 정보
                source_info = {
                    "kb_type": kb_type,
                    "language": language,
                    "text": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    "score": node.score if hasattr(node, 'score') else None,
                    "source": metadata.get("source", "Unknown"),
                    "doc_id": metadata.get("doc_id", "Unknown"),
                }
                
                # 지식베이스 타입별 메타데이터 추출
                if kb_type == 'lecture_notes':
                    source_info.update({
                        "lecture_title": metadata.get("lecture_title", "Unknown"),
                        "page_no": metadata.get("page_no", "N/A"),
                        "has_tables": metadata.get("has_tables", False),
                        "has_images": metadata.get("has_images", False),
                        "has_script": metadata.get("has_script", False),
                    })
                
                elif kb_type == 'textbook':
                    source_info.update({
                        "chapter_id": metadata.get("chapter_id", "Unknown"),
                        "chapter_title": metadata.get("chapter_title", "Unknown"),
                        "sub_chapter_id": metadata.get("sub_chapter_id", ""),
                        "sub_chapter_title": metadata.get("sub_chapter_title", ""),
                    })
                
                sources.append(source_info)
            
            # 출처별 개수 계산
            lecture_count = sum(1 for s in sources if s['kb_type'] == 'lecture_notes')
            textbook_count = sum(1 for s in sources if s['kb_type'] == 'textbook')
            
            result = {
                "answer": str(response),
                "sources": sources,
                "source_summary": {
                    "lecture_notes": lecture_count,
                    "textbook": textbook_count,
                    "total": len(sources)
                },
                "success": True,
                "mode": self._get_mode_name()
            }
            return result
            
        except Exception as e:
            print(f"답변 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "source_summary": {"lecture_notes": 0, "textbook": 0, "total": 0},
                "success": False,
                "mode": self._get_mode_name()
            }
    
    def _query_llm_only(self, question: str) -> Dict:
        prompt = f"""당신은 열역학 과목을 가르치는 AI 조교입니다. 다음 질문에 대해 답변해주세요.

질문: {question}

답변 시 다음 사항을 준수하세요:
1. 열역학 개념과 원리를 정확하게 설명하세요
2. 필요한 경우 수식을 포함하여 설명하세요
3. 명확하고 체계적으로 답변하세요
4. 확신이 없는 경우 "확실하지 않습니다"라고 명시하세요

답변:"""
        
        try:
            response = self.llm.complete(prompt)
            
            return {
                "answer": str(response),
                "sources": [],
                "source_summary": {"lecture_notes": 0, "textbook": 0, "total": 0},
                "success": True,
                "mode": "llm_only"
            }
        except Exception as e:
            return {
                "answer": f"LLM 응답 생성 실패: {str(e)}",
                "sources": [],
                "source_summary": {"lecture_notes": 0, "textbook": 0, "total": 0},
                "success": False,
                "mode": "llm_only"
            }
    
    def _get_mode_name(self) -> str:
        if not self.use_lecture and not self.use_textbook:
            return "llm_only"
        elif self.use_lecture and not self.use_textbook:
            return "lecture_only"
        elif not self.use_lecture and self.use_textbook:
            return "textbook_only"
        else:
            return "dual"