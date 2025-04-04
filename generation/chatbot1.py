import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_response(user_message):
    # (1) 시스템 프롬프트 정의 (원하는 설정 유지)
    system_prompt = (
        "너의 이름은 산타 침착맨이야. 반말을 사용하고 나에게 매우 장난스럽게 얘기해. "
        "산타와 관련된 농담이나 유머를 적극 활용해."
    )
    
    # (2) OpenAI Chat API 호출 (간단 버전)
    response = client.chat.completions.create(
        model="gpt-4o",  # 실제 사용하는 모델명에 맞게 조정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    
    # (3) 응답 추출
    reply = response.choices[0].message.content
    return reply
