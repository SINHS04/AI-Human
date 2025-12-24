import yaml

from src.dual_llm_config import DualLLMConfigurator
from src.dual_vector_store import DualVectorStoreManager
from src.dual_rag_engine import DualRAGEngine

config_path = "./config/datastructure_config.yaml"

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 지식베이스 선택 설정
kb_selection = config.get('knowledge_base_selection', {})
use_lecture = kb_selection.get('use_lecture_notes', True)
use_textbook = kb_selection.get('use_textbook', True)

# LLM 로드 (항상 필요)
configurator = DualLLMConfigurator(config_path)
llm = configurator.setup_llm()

# 변수 초기화
korean_embed = None
english_embed = None
lecture_index = None
textbook_index = None
lecture_nodes = []
textbook_nodes = []

# 필요한 경우에만 임베딩 모델 및 인덱스 로드
if use_lecture or use_textbook:
    korean_embed, english_embed = configurator.setup_embedding_models()
    
    vector_manager = DualVectorStoreManager(config_path)
    
    # 강의자료 로드
    if use_lecture:
        lecture_index, _ = vector_manager.load_dual_indexes(
            korean_embed, english_embed
        )
        lecture_nodes = vector_manager.get_all_nodes(lecture_index)
        print(f"강의자료 로드: {len(lecture_nodes)}개 노드")
    
    # 교재 로드
    if use_textbook:
        _, textbook_index = vector_manager.load_dual_indexes(
            korean_embed, english_embed
        )
        textbook_nodes = vector_manager.get_all_nodes(textbook_index)
        print(f"교재 로드: {len(textbook_nodes)}개 노드")

# RAG 엔진 생성
rag_engine = DualRAGEngine(
    lecture_index, textbook_index,
    lecture_nodes, textbook_nodes,
    llm, korean_embed, english_embed,
    config_path
)

# 터미널에서 종료할때까지 input을 받고, output 출력
while True:
    try:
        user_input = input("\n질문 입력 (종료는 exit): ")
        
        if user_input.lower() in ['exit']:
            print("챗봇을 종료합니다.")
            break
        
        response = rag_engine.query(user_input)
        
        print(f"\n챗봇 응답:\n{response['answer']}\n")
    
    except KeyboardInterrupt:
        print("\n챗봇을 종료합니다.")
        break