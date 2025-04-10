import os 
from openai import OpenAI 
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv

# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
# do it
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=api_key) 

# 매 실행 시 DB 폴더를 삭제 후 새로 생성
def init_db(db_path="./static/data/chatbot2/chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    # collection = dbclient.create_collection(name="rag_collection", get_or_create=True) # 이 줄 대신 아래 줄 사용
    collection = dbclient.get_or_create_collection(name="rag_collection") # 이게 더 표준적이고 안전
    return dbclient, collection


# 텍스트 로딩 함수
def load_text_files(folder_path):
    # do it
    docs=[]
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename) 
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append((filename, text))
    return docs


# OpenAI Embeddings 생성 함수 
def get_embedding(text, model="text-embedding-3-large"):
    try:
        # text가 비어있는 경우 처리 추가
        processed_text = text.strip()
        if not processed_text:
            print("Warning: Empty text received for embedding.")
            return None # 또는 빈 벡터 반환? None이 나을 수 있음
        response = client.embeddings.create(input=[processed_text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding for text '{text[:50]}...': {e}")
        return None # 실패 시 None 반환


# # 문서 청크 단위로 나누기
# def chunk_text(text, chunk_size=400, chunk_overlap=50):
#     # do it 
#     chunks=[]
#     start=0
#     while start<len(text):
#         end=start+chunk_size
#         chunk=text[start:end]
#         chunks.append(chunk)
#         start=end-chunk_overlap

#         if start<0:
#             start=0

#         if start>=len(text):
#             break

#     return chunks


from langchain.text_splitter import RecursiveCharacterTextSplitter

# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":

    # db 초기화, 경로 지정
    dbclient, collection = init_db("./static/data/chatbot2/chroma_db")

    folder_path = "./static/data/chatbot2"
    # load_text_files 함수로 처리할 문서 데이터 불러오기기
    docs = load_text_files(folder_path)
    # do it

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
    )

    doc_id = 0
    for filename, text in docs: 
        chunks = text_splitter.split_text(text) # chunking
        for idx, chunk in enumerate(chunks): # 각 청크와 해당 청크의 인덱스 가져옴
            doc_id += 1 # 인덱스 하나씩 증가 시키면서
            embedding = get_embedding(chunk) # 각 청크 임베딩 벡터 생성
            # vectorDB에 다음 정보 추가
            if "food" in filename.lower():  # 파일명이 'food' 포함하면 음식 관련으로 판단
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": filename,
                        "chunk_index": idx,
                        "image_url": f"/static/images/chatbot2/{filename.replace('.txt', '.png')}",
                        "emotion": "happy"
                    }],
                    ids=[str(doc_id)]
                )
            else:
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "filename": filename,
                        "chunk_index": idx
                    }],
                    ids=[str(doc_id)]
                )
    # 전처리 과정 
    # do it

    print("모든 문서 벡터DB에 저장 완료")