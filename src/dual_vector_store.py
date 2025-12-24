import os
from typing import List, Dict, Tuple
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.embeddings import BaseEmbedding
import chromadb
import yaml


class DualVectorStoreManager:
    def __init__(self, config_path: str = "./config/thermodynamics_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.vector_configs = self.config['vector_stores']
        self.lecture_config = self.vector_configs['lecture_notes']
        self.textbook_config = self.vector_configs['textbook']
    
    def create_vector_store(self, vs_config: Dict, embed_model: BaseEmbedding) -> Tuple[VectorStoreIndex, ChromaVectorStore]:
        persist_dir = vs_config['persist_directory']
        collection_name = vs_config['collection_name']
        
        os.makedirs(persist_dir, exist_ok=True)
        
        print(f"\n벡터 스토어 생성: {collection_name}")
        
        # Chroma 클라이언트
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        print(f"벡터 스토어 생성 완료: {collection_name}\n")
        return chroma_client, vector_store
    
    def create_dual_indexes(
        self,
        lecture_docs: List[Document],
        textbook_docs: List[Document],
        korean_embed: BaseEmbedding,
        english_embed: BaseEmbedding,
        force_rebuild: bool = False
    ) -> Tuple[VectorStoreIndex, VectorStoreIndex]:        
        # 강의자료 인덱스
        print("\n강의자료 벡터 인덱스 생성")
        lecture_client, lecture_vector_store = self.create_vector_store(
            self.lecture_config, korean_embed
        )
        
        lecture_docstore = SimpleDocumentStore()
        lecture_storage_context = StorageContext.from_defaults(
            vector_store=lecture_vector_store,
            docstore=lecture_docstore
        )
        
        lecture_index = VectorStoreIndex.from_documents(
            lecture_docs,
            storage_context=lecture_storage_context,
            embed_model=korean_embed,
            show_progress=True
        )
        
        lecture_storage_context.persist(persist_dir=self.lecture_config['persist_directory'])
        print(f"강의자료 인덱스 생성 완료: {len(lecture_docs)}개 문서\n")
        
        # 교재 인덱스
        print("\n교재 벡터 인덱스 생성")
        textbook_client, textbook_vector_store = self.create_vector_store(
            self.textbook_config, english_embed
        )
        
        textbook_docstore = SimpleDocumentStore()
        textbook_storage_context = StorageContext.from_defaults(
            vector_store=textbook_vector_store,
            docstore=textbook_docstore
        )
        
        textbook_index = VectorStoreIndex.from_documents(
            textbook_docs,
            storage_context=textbook_storage_context,
            embed_model=english_embed,
            show_progress=True
        )
        
        textbook_storage_context.persist(persist_dir=self.textbook_config['persist_directory'])
        print(f"교재 인덱스 생성 완료: {len(textbook_docs)}개 문서\n")
        
        return lecture_index, textbook_index
    
    def load_dual_indexes(
        self,
        korean_embed: BaseEmbedding,
        english_embed: BaseEmbedding
    ) -> Tuple[VectorStoreIndex, VectorStoreIndex]:
        print("\n강의자료 인덱스 로드 중...")
        _, lecture_vector_store = self.create_vector_store(self.lecture_config, korean_embed)
        lecture_storage_context = StorageContext.from_defaults(
            vector_store=lecture_vector_store,
            persist_dir=self.lecture_config['persist_directory']
        )
        lecture_index = VectorStoreIndex.from_vector_store(
            vector_store=lecture_vector_store,
            storage_context=lecture_storage_context,
            embed_model=korean_embed
        )
        print("강의자료 인덱스 로드 완료\n")
        
        print("\n교재 인덱스 로드 중...")
        _, textbook_vector_store = self.create_vector_store(self.textbook_config, english_embed)
        textbook_storage_context = StorageContext.from_defaults(
            vector_store=textbook_vector_store,
            persist_dir=self.textbook_config['persist_directory']
        )
        textbook_index = VectorStoreIndex.from_vector_store(
            vector_store=textbook_vector_store,
            storage_context=textbook_storage_context,
            embed_model=english_embed
        )
        print("교재 인덱스 로드 완료\n")
        
        return lecture_index, textbook_index
    
    def get_all_nodes(self, index: VectorStoreIndex) -> List:
        try:
            collection = index.vector_store._collection
            result = collection.get(include=['documents', 'metadatas'])
            
            from llama_index.core.schema import TextNode
            nodes = []
            for i, doc_id in enumerate(result['ids']):
                node = TextNode(
                    id_=doc_id,
                    text=result['documents'][i] if result['documents'] else "",
                    metadata=result['metadatas'][i] if result['metadatas'] else {}
                )
                nodes.append(node)
            
            return nodes
        except Exception as e:
            print(f"노드 가져오기 실패: {e}")
            return []