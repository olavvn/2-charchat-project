import os
import json
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import logging

# --- 기본 설정 ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 경로 설정 (수정됨) ---
# 입력 JSON 파일 경로 변경
JSON_MAP_PATH = os.path.join('static', 'images', 'chatbot2', 'chatbot2_food_images.json')
OUTPUT_DIR = os.path.join("static", "data", "chatbot2")
# 출력 pkl 파일 이름 변경 (기존 파일 덮어쓰기 방지)
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, "food_emotion_embeddings.pkl")

# --- OpenAI 클라이언트 초기화 ---
# ... (이전 스크립트와 동일) ...
api_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if not api_key:
    logging.error("CRITICAL: OPENAI_API_KEY environment variable not set.")
else:
    try:
        openai_client = OpenAI(api_key=api_key)
        logging.info("OpenAI client initialized successfully.")
    except Exception as e:
        logging.error(f"CRITICAL: Error initializing OpenAI client: {e}")

# --- 임베딩 함수 정의 ---
# ... (이전 스크립트와 동일) ...
def get_embedding(text, model="text-embedding-3-large"):
    if not openai_client or not text: return None
    try:
        processed_text = text.strip().replace("\n", " ")
        if not processed_text: return None
        response = openai_client.embeddings.create(input=[processed_text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logging.warning(f"Warning: Error getting embedding for text '{text[:50]}...': {e}")
        return None

# --- 메인 실행 로직 ---
def main():
    logging.info(f"Starting embedding generation process for food emotions...")
    logging.info(f"Input JSON: {JSON_MAP_PATH}")
    logging.info(f"Output PKL: {OUTPUT_FILE_PATH}")

    if not openai_client:
        logging.error("OpenAI client is not available. Cannot proceed.")
        return

    # --- 1. JSON 맵 로드 (구조 변경 반영) ---
    emotion_data_map = {} # 이름 변경 (이제 단순 map이 아님)
    try:
        with open(JSON_MAP_PATH, 'r', encoding='utf-8') as f:
            # JSON 파일의 최상위 객체 자체가 감정 키를 가짐
            all_data = json.load(f)
            if isinstance(all_data, dict):
                # all_data가 { "감정키": { "image": ..., "message": ... } } 형태라고 가정
                emotion_data_map = all_data
                logging.info(f"Successfully loaded food emotion data from '{JSON_MAP_PATH}'. Found {len(emotion_data_map)} emotions.")
            else:
                logging.error(f"ERROR: Expected a dictionary at the top level in '{JSON_MAP_PATH}'.")
                return
    # ... (파일 로드 관련 오류 처리 동일) ...
    except FileNotFoundError:
        logging.error(f"ERROR: Food emotion JSON file not found at '{JSON_MAP_PATH}'.")
        return
    except json.JSONDecodeError:
        logging.error(f"ERROR: Failed to decode JSON from '{JSON_MAP_PATH}'. Check format.")
        return
    except Exception as e:
        logging.error(f"ERROR: An unexpected error occurred loading map: {e}.")
        return

    if not emotion_data_map:
        logging.warning("Food emotion data map is empty. No labels to process.")
        return

    # --- 2. 임베딩 계산 ---
    food_emotion_label_embeddings = {} # 저장할 딕셔너리 이름 변경
    # 감정 키워드(최상위 키)를 가져와서 임베딩 계산
    # '기본' 같은 특정 키를 제외할 필요가 있는지 확인 (이 JSON 구조에서는 없을 수 있음)
    labels_to_process = list(emotion_data_map.keys())
    logging.info(f"Calculating embeddings for {len(labels_to_process)} emotion labels...")
    calculated_count = 0

    for label in labels_to_process:
        logging.info(f"Processing label: '{label}'")
        embedding = get_embedding(label) # 감정 키워드 자체를 임베딩
        if embedding:
            food_emotion_label_embeddings[label] = np.array(embedding)
            calculated_count += 1
            logging.info(f"  -> Embedding calculated successfully.")
        else:
            logging.warning(f"  -> Failed to calculate embedding for '{label}'. It will be excluded.")

    logging.info(f"Finished calculation. Successfully processed {calculated_count} labels.")

    # --- 3. 계산된 임베딩 파일 저장 ---
    if food_emotion_label_embeddings:
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logging.info(f"Ensured output directory exists: {OUTPUT_DIR}")
            with open(OUTPUT_FILE_PATH, 'wb') as f:
                # 새 딕셔너리 저장
                pickle.dump(food_emotion_label_embeddings, f)
            logging.info(f"Successfully saved calculated food emotion embeddings to: {OUTPUT_FILE_PATH}")
        except Exception as e:
            logging.error(f"Error saving calculated embeddings to file '{OUTPUT_FILE_PATH}': {e}")
    else:
        logging.warning("No embeddings were successfully calculated. Nothing to save.")

    logging.info("Food emotion embedding generation process finished.")

if __name__ == "__main__":
    main()