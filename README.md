# 자료구조 Q&A 챗봇
### 이중 벡터 DB 기반 RAG 시스템

### 설치
```
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env
```

### 사용법
1. 벡터 인덱스 빌드
```
python build_dual_index.py --config ./config/datastructure_config.yaml
```

2. 챗봇 실행
```
python run.py
```

### 입출력 명세
1. 입력
```
user_input: str
```

2. 출력
```
response: Dict = {
    "answer": str,          # LLM 생성 답변
    "sources": List[Dict],  # 검색된 문서 리스트
    "source_summary": {
        "lecture_notes": int,  # 강의자료 개수
        "textbook": int,       # 교재 개수
        "total": int          # 전체 개수
    },
    "success": bool,        # 성공 여부
    "mode": str            # "dual" | "lecture_only" | "textbook_only" | "llm_only"
}
```

### 시스템 최적화
1. config.yaml 파일
    - llm_model: name > 원하는 모델 선택(google)
    - rag: similarity_top_k > 문석 개수 설정
2. prompt 파일
    - qa_default.txt > 원하는 답변이 나오도록 프롬프트 구성