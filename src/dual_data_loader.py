import json
from typing import List, Dict, Tuple
from pathlib import Path
from llama_index.core import Document
import yaml


class DualDataLoader:
    def __init__(self, config_path: str = "./config/thermodynamics_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.data_config = config['data']
        self.lecture_config = self.data_config['lecture_notes']
        self.textbook_config = self.data_config['textbook']
        
        self.lecture_data = []
        self.textbook_data = []
    
    def load_knowledge_base(self, kb_config: Dict) -> List[Dict]:
        kb_dir = Path(kb_config['knowledge_base_dir'])
        file_pattern = kb_config.get('file_pattern', '**/*.json')
        exclude_patterns = kb_config.get('exclude_patterns', [])
        
        if not kb_dir.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {kb_dir}")
        
        # JSON 파일 찾기
        all_files = list(kb_dir.glob(file_pattern))
        
        # 제외 패턴 필터링
        filtered_files = []
        for file_path in all_files:
            should_exclude = False
            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        
        print(f"\n{kb_config['knowledge_base_dir']} - {len(filtered_files)}개 파일")
        
        # JSON 파일 로드
        all_data = []
        for file_path in filtered_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    data = [data]
                
                # 메타데이터 추가
                for item in data:
                    if not item.get('text'):
                        continue
                    
                    if 'metadata' not in item:
                        item['metadata'] = {}
                    
                    item['metadata']['file_path'] = str(file_path)
                    item['metadata']['file_name'] = file_path.name
                    item['metadata']['directory'] = str(file_path.parent.relative_to(kb_dir))
                    item['metadata']['language'] = kb_config.get('language', 'unknown')
                    
                    all_data.append(item)
                    
            except Exception as e:
                print(f"파일 로드 오류 ({file_path}): {e}")
                continue
        
        print(f"데이터 로드 완료: {len(all_data)}개 문서\n")
        return all_data
    
    def load_all_knowledge_bases(self) -> Tuple[List[Dict], List[Dict]]:
        print("\n### 강의자료 로딩 ###")
        self.lecture_data = self.load_knowledge_base(self.lecture_config)
        
        print("\n### 교재 로딩 ###")
        self.textbook_data = self.load_knowledge_base(self.textbook_config)
        
        return self.lecture_data, self.textbook_data
    
    def create_documents(self, data: List[Dict], kb_type: str) -> List[Document]:
        documents = []
        skipped_count = 0
        
        for item in data:
            text = item.get("text", "").strip()
            if len(text) < 10:
                skipped_count += 1
                continue
            
            # 원본 메타데이터 가져오기
            original_metadata = item.get("meta_data", {})
            
            # 기본 메타데이터 구성
            metadata = {
                "kb_type": kb_type,
                "doc_id": item.get("id", ""),  # 'id'를 'doc_id'로 변경 (충돌 방지)
                "source": item.get("source", "Unknown"),
                "type": item.get("type", ""),
            }
            
            # 지식 베이스 타입별 메타데이터 추가
            if kb_type == 'lecture_notes':
                # 강의자료 메타데이터
                metadata.update({
                    "lecture_title": original_metadata.get("lecture_title", "Unknown"),
                    "page_no": original_metadata.get("page_no", 0),
                    "has_tables": len(original_metadata.get("tables", [])) > 0,
                    "has_images": len(original_metadata.get("images", [])) > 0,
                    "has_script": bool(original_metadata.get("script", "").strip()),
                })
                
                # 디버깅 출력 (첫 3개만)
                if len(documents) < 3:
                    print(f"  [강의자료 샘플] lecture_title: {metadata['lecture_title']}, page_no: {metadata['page_no']}")
            
            elif kb_type == 'textbook':
                # 교재 메타데이터
                metadata.update({
                    "chapter_id": original_metadata.get("chapter_id", "Unknown"),
                    "chapter_title": original_metadata.get("chapter_title", "Unknown"),
                    "sub_chapter_id": original_metadata.get("sub_chapter_id", ""),
                    "sub_chapter_title": original_metadata.get("sub_chapter_title", ""),
                })
                
                # 디버깅 출력 (첫 3개만)
                if len(documents) < 3:
                    print(f"  [교재 샘플] chapter_id: {metadata['chapter_id']}, chapter_title: {metadata['chapter_title']}")
            
            try:
                doc = Document(
                    text=text,
                    metadata=metadata,
                    doc_id=item.get("id", f"{kb_type}_{len(documents)}")
                )
                documents.append(doc)
            except Exception as e:
                print(f"⚠️  Document 생성 실패: {e}")
                skipped_count += 1
        
        print(f"{kb_type} Document 생성 완료: {len(documents)}개")
        if skipped_count > 0:
            print(f"건너뛴 항목: {skipped_count}개")
        
        return documents
    
    def get_statistics(self) -> Dict:
        return {
            "lecture_notes": {
                "total_documents": len(self.lecture_data),
                "language": self.data_config['lecture_notes'].get('language', 'unknown')
            },
            "textbook": {
                "total_documents": len(self.textbook_data),
                "language": self.data_config['textbook'].get('language', 'unknown')
            },
            "total_documents": len(self.lecture_data) + len(self.textbook_data)
        }