import os
import sys
import subprocess
from flask import Flask, request, render_template, jsonify, url_for, session # session 추가
from dotenv import load_dotenv
# app.py 상단에 추가
from flask_cors import CORS # CORS 추가하신 것 유지

load_dotenv() # .env 파일 로드 먼저 수행

app = Flask(__name__)

# ***** Secret Key 설정 *****
# .env 파일에서 FLASK_SECRET_KEY 값을 읽어와 설정합니다.
# 값이 없다면 기본값(개발용으로만 사용!)을 사용하거나 오류를 발생시킬 수 있습니다.
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    print("경고: FLASK_SECRET_KEY가 .env 파일에 설정되지 않았습니다. 임시 키를 사용합니다. (운영 환경에서는 절대 사용 금지)")
    # 개발 중 임시 키 (절대 운영 환경에서 사용하지 마세요)
    app.secret_key = 'dev-secret-key-replace-this-later'
    # 또는 raise ValueError("FLASK_SECRET_KEY 환경 변수가 설정되지 않았습니다.") 로 설정 강제

CORS(app) # CORS 설정 (Secret Key 설정 이후에 해도 무방)

MAX_SESSION_HISTORY_LENGTH = 15

# 초기 화면: 챗봇 선택 페이지
@app.route('/')
def index():
    chatbots = [
        {
            'id': 1,
            'name': 'chatbot1',
            'image': url_for('static', filename='images/chatbot1/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        },
        {
            'id': 2,
            'name': '은하수 식당',
            'image': url_for('static', filename='images/chatbot2/gallery01.png'),
            # chatbot2 태그 수정하신 것 반영
            'tags': ['#다정남', '#상담캐', '#에겐남', '#미중년', '#존잘']
        },
        {
            'id': 3,
            'name': 'chatbot3',
            'image': url_for('static', filename='images/chatbot3/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        },
        {
            'id': 4,
            'name': 'chatbot4',
            'image': url_for('static', filename='images/chatbot4/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        }
    ]
    return render_template('index.html', chatbots=chatbots)

# 챗봇 상세정보 페이지 (새로운 HTML로 구현)
@app.route('/detail/<int:bot_id>')
def detail(bot_id):
    chatbot_data = {
        1: {
            "name": "chatbot1",
            'image': url_for('static', filename='images/chatbot1/thumbnail.png'),
            "description": "chatbot1의 설명입니다.",
            'tags': ['#챗봇', '#유머', '#일상']
        },
        2: {
            "name": "은하수 식당",
            'image': url_for('static', filename='images/chatbot2/gallery01.png'),
            "description": "어서오세요, 밤에만 볼 수 있는 은하수 식당입니다.",
             # detail 페이지 태그에도 멀티턴 기능 추가 (선택 사항)
            'tags': ['#다정남', '#상담캐', '#에겐남', '#미중년', '#존잘']
        },
        3: {
            "name": "chatbot3",
            'image': url_for('static', filename='images/chatbot3/thumbnail.png'),
            "description": "chatbot3의 설명입니다.",
            'tags': ['#챗봇', '#유머', '#일상']
        },
        4: {
            "name": "chatbot4",
            'image': url_for('static', filename='images/chatbot4/thumbnail.png'),
            "description": "chatbot4의 설명입니다.",
            'tags': ['#챗봇', '#유머', '#일상']
        },
    }
    bot = chatbot_data.get(bot_id)
    if not bot:
        return "Invalid bot id", 404
    return render_template('chatbot_detail.html', bot=bot, bot_id=bot_id)

# 공용 채팅 화면: URL의 bot_id에 따라 제목 등을 변경 (변경 없음)
# 참고: 채팅 화면 진입 시 이전 기록을 지우고 싶다면 여기서 session.pop(f'history_{bot_id}', None) 호출 가능
@app.route('/chat/<int:bot_id>')
def chat(bot_id):
    username = request.args.get("username", "손")
    # --- 추가된 부분: 채팅 페이지 로드 시 해당 봇의 히스토리 초기화 ---
    session_key = f'history_{bot_id}'
    if session_key in session:
        session.pop(session_key, None)  # 해당 bot_id의 history를 세션에서 제거
        session.modified = True # 세션 변경 사항 저장 요청 (pop 이후에도 명시하는 것이 안전)
        print(f"DEBUG [app.py]: Cleared chat history for bot_id {bot_id} because chat page was loaded.")
    # --- 추가된 부분 끝 ---
    chatbot_names = {
        1: "chatbot1",
        2: "은하수 식당",
        3: "chatbot3",
        4: "chatbot4"
    }
    bot_name = chatbot_names.get(bot_id, "챗봇")
    # 채팅 페이지 로드 시 이전 대화 기록을 클라이언트에 전달 (선택 사항)
    # 이 부분은 현재 JS에서 사용하지 않으므로 주석 처리해도 무방하나,
    # 페이지 로드 시 이전 대화를 바로 보여주고 싶다면 유지하고 chat.html에서 사용
    history = session.get(f'history_{bot_id}', [])
    return render_template('chat.html', bot_id=bot_id, bot_name=bot_name, history=history, username=username)


# ***** 수정된 API 엔드포인트 *****
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must be JSON'}), 400

    user_message = data.get('message')
    bot_id_str = data.get('bot_id') # 우선 문자열로 받고 검증

    if not user_message:
        return jsonify({'error': 'Missing message'}), 400
    if not bot_id_str:
         return jsonify({'error': 'Missing bot_id'}), 400

    try:
        bot_id = int(bot_id_str)
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid bot id format'}), 400

    # ***** 세션에서 해당 봇의 대화 기록 가져오기 *****
    session_key = f'history_{bot_id}'
    # 세션 초기화 확인 (만약 세션 키가 처음 생성되는 경우 빈 리스트 할당)
    if session_key not in session:
        session[session_key] = []
        # print(f"DEBUG: Initialized empty history for {session_key}") # 디버깅용 로그

    conversation_history = session[session_key] # 이제 안전하게 접근 가능
    print(f"DEBUG [app.py]: Passing history with length {len(conversation_history)} to chatbot {bot_id}")

    # print(f"DEBUG: Loaded history for {session_key}: {len(conversation_history)} messages") # 디버깅용 로그


    try:
        # 챗봇 모듈 로드 및 함수 호출 (대화 기록 전달)
        generate_response = None # 함수 초기화
        if bot_id == 1:
            from generation.chatbot1 import generate_response
        elif bot_id == 2:
            from generation.chatbot2 import generate_response
        elif bot_id == 3:
            from generation.chatbot3 import generate_response
        elif bot_id == 4:
            from generation.chatbot4 import generate_response
        else:
            return jsonify({'error': 'Invalid bot id'}), 400

        # generate_response 함수가 실제로 로드되었는지 확인
        if generate_response is None:
             # 위 if/elif 에서 해당 bot_id에 맞는 모듈을 찾지 못한 경우 발생 가능성은 낮음
             return jsonify({'error': f'Could not load generate_response for bot_id {bot_id}'}), 500

        # 중요: 각 chatbot 모듈의 generate_response 함수는
        # user_message와 conversation_history 두 개의 인자를 받도록 수정되어야 합니다.
        reply = generate_response(user_message, conversation_history)
        # ***** 디버깅 출력 추가 *****
        print(f"DEBUG: Type of reply received from generate_response: {type(reply)}")
        print(f"DEBUG: Value of reply received: {reply}")
        # ***** 디버깅 출력 끝 *****
        reply_text_only = ""
        if isinstance(reply, dict) and 'reply' in reply:
            reply_text_only = reply['reply'] # 딕셔너리에서 'reply' 값(문자열)만 추출
        elif isinstance(reply, str): # 혹시 문자열만 반환된 경우
            reply_text_only = reply

        # ***** 대화 기록 업데이트 및 세션에 저장 *****
        # conversation_history는 이제 session[session_key]와 같은 객체를 참조함
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": reply_text_only})
        print(f"DEBUG [app.py]: History length after update: {len(conversation_history)}")


        # 기록 길이 제한 적용 (맨 앞의 오래된 기록부터 제거)
        while len(conversation_history) > MAX_SESSION_HISTORY_LENGTH * 2:
             conversation_history.pop(0) # 맨 앞(가장 오래된) user 메시지 제거
             conversation_history.pop(0) # 맨 앞(가장 오래된) assistant 메시지 제거
             # print(f"DEBUG: History trimmed for {session_key}") # 디버깅용 로그

        # session[session_key] = conversation_history # 이미 같은 리스트 객체이므로 재할당 불필요
        session.modified = True # 세션이 변경되었음을 명시 (리스트 등 mutable 타입 수정 시 필수!)

        return jsonify(reply)

    except ModuleNotFoundError:
        print(f"ERROR: Chatbot module for bot_id {bot_id} not found.")
        return jsonify({'error': f'Chatbot module for bot_id {bot_id} not found.'}), 500
    except AttributeError:
        # 함수가 없거나, history 인자를 받지 않는 경우 발생 가능
        print(f"ERROR: generate_response function signature error in module for bot_id {bot_id}.")
        import traceback
        traceback.print_exc() # 어떤 함수 시그니처 문제인지 상세 로그 확인
        return jsonify({'error': f'Chatbot function signature error for bot_id {bot_id}. Make sure it accepts history.'}), 500
    except TypeError as e:
         # generate_response 함수 호출 시 인자 개수 등이 맞지 않을 때 발생 가능
         print(f"ERROR calling generate_response for bot_id {bot_id}: {e}")
         import traceback
         traceback.print_exc()
         return jsonify({'error': f'Error calling chatbot function for bot_id {bot_id}: {e}'}), 500
    except Exception as e:
        print(f"ERROR in /api/chat for bot_id {bot_id}: {e}") # 서버 로그에 상세 에러 기록
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {e}'}), 500



if __name__ == '__main__':
    vector_db_script_path = os.path.join('generation', 'build_vector_db_chatbot2.py')
    print(f"Attempting to run vector DB build script: {vector_db_script_path}")

    try:
        # subprocess를 사용하여 build_vector_db_chatbot2.py 실행
        result = subprocess.run(
            [sys.executable, vector_db_script_path],
            check=True, # 오류 발생 시 예외 발생
            capture_output=True,
            text=True
        )
        print("Vector DB build script executed successfully.")
        print("Script output:\n", result.stdout)
    except FileNotFoundError:
        print(f"Error: The script was not found at {vector_db_script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing vector DB build script (Return code: {e.returncode}):")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while trying to run the script: {e}")

    # Flask 개발 서버 실행
    print("Starting Flask development server...")
    app.run(debug=True)