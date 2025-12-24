import argparse
from src.dual_data_loader import DualDataLoader
from src.dual_llm_config import DualLLMConfigurator
from src.dual_vector_store import DualVectorStoreManager


def main():
    parser = argparse.ArgumentParser(description="벡터 인덱스 빌드")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/datastructure_config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="기존 인덱스를 삭제하고 새로 생성"
    )
    
    args = parser.parse_args()
    
    print("#" * 60)
    print("이중 벡터 인덱스 빌드 시작")
    print("#" * 60)
    
    try:
        # 1. 데이터 로드
        print("\n[1/4] 데이터 로딩...")
        loader = DualDataLoader(args.config)
        lecture_data, textbook_data = loader.load_all_knowledge_bases()
        
        # 통계 출력
        stats = loader.get_statistics()
        print(f"데이터 통계:")
        print(f"  - 강의자료: {stats['lecture_notes']['total_documents']:,}개 ({stats['lecture_notes']['language']})")
        print(f"  - 교재: {stats['textbook']['total_documents']:,}개 ({stats['textbook']['language']})")
        print(f"  - 총 문서 수: {stats['total_documents']:,}개")
        
        # 2. 모델 로드
        print("\n[2/4] 임베딩 모델 로딩...")
        configurator = DualLLMConfigurator(args.config)
        korean_embed, english_embed = configurator.setup_embedding_models()
        
        # 3. Document 생성
        print("\n[3/4] Document 객체 생성...")
        lecture_docs = loader.create_documents(lecture_data, 'lecture_notes')
        textbook_docs = loader.create_documents(textbook_data, 'textbook')
        
        # 4. 벡터 인덱스 생성
        print("\n[4/4] 이중 벡터 인덱스 생성...")
        print("이 작업은 시간이 걸릴 수 있습니다...")
        
        vector_manager = DualVectorStoreManager(args.config)
        lecture_index, textbook_index = vector_manager.create_dual_indexes(
            lecture_docs,
            textbook_docs,
            korean_embed,
            english_embed,
            force_rebuild=args.rebuild
        )
        
        # 완료
        print("\n" + "#" * 60)
        print("이중 벡터 인덱스 빌드 완료")
        print("#" * 60)
        
        print(f"\n인덱스 정보:")
        print(f"  - 강의자료: {len(lecture_docs):,}개 문서 (한국어 임베딩)")
        print(f"  - 교재: {len(textbook_docs):,}개 문서 (영어 임베딩)")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())