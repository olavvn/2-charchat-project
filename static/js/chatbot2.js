// 파일 경로: static/js/chatbot2.js

// chat-area에서 bot_id 정보 꺼내기
const chatArea = document.querySelector('.chat-area');
const botId = chatArea.dataset.botId; 

// 주요 DOM 요소
const chatLog = document.getElementById('chat-log');
const userMessageInput = document.getElementById('user-message');
const sendBtn = document.getElementById('send-btn');
const videoBtn = document.getElementById('videoBtn'); 
const imageBtn = document.getElementById('imageBtn'); 

// --- 메시지 전송 함수 ---
async function sendMessage() {
  const message = userMessageInput.value.trim();
  if (!message || !botId) {
      console.error("메시지가 없거나 botId가 유효하지 않습니다.");
      return;
  }

  // 사용자 메시지 표시 (여기서의 message는 .trim()을 거쳐 항상 문자열)
  appendMessage('user', message, null); 
  userMessageInput.value = '';
  // Optional: show loading indicator here

  try {
      // API 호출 (conversation_history 없이)
      const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
              bot_id: parseInt(botId), 
              message: message 
          }) 
      });

      // Optional: hide loading indicator here

      if (!response.ok) {
          let errorData = { error: `서버 응답 오류: ${response.status}` };
          try { errorData = await response.json(); } catch (e) {}
          console.error("Server error:", errorData);
          // 오류 메시지도 appendMessage로 표시
          appendMessage('bot', `Error: ${errorData.error || response.statusText}`, null);
          return;
      }

      const data = await response.json();
      console.log("DEBUG: 서버로부터 받은 전체 데이터:", JSON.stringify(data, null, 2));

      // ***** 수정된 서버 응답 처리 *****
      // data.reply가 존재하고, 그것이 객체인지 확인
      if (data.reply && typeof data.reply === 'object') { 
        const nestedReplyData = data.reply; // 중첩된 객체에 접근

        // 중첩된 객체에서 실제 텍스트와 이미지 URL 추출
        const botReplyText = nestedReplyData.reply ?? ''; // 중첩 객체 안의 reply
        const botImageUrl = nestedReplyData.image_url; // 중첩 객체 안의 image_url

        // 실제 텍스트나 이미지 URL이 있을 경우에만 메시지 추가
        if (botReplyText || botImageUrl) {
             appendMessage('bot', botReplyText, botImageUrl); 
        } else {
             // 중첩 객체는 있지만 내용이 비어있는 경우 처리 (선택적)
             console.error("Received nested reply object, but it's empty or missing data:", nestedReplyData);
             appendMessage('bot', 'Error: 서버로부터 유효한 응답 데이터를 받지 못했습니다.', null);
        }
        
      } else if (data.error) {
          console.error("Server returned error:", data.error);
          appendMessage('bot', 'Error: ' + data.error, null);
      } else {
          console.error("Unexpected response format:", data);
          appendMessage('bot', 'Error: 서버로부터 예기치 않은 응답을 받았습니다.', null);
      }

  } catch (err) {
      // Optional: hide loading indicator here
      console.error("Fetch Error:", err);
      appendMessage('bot', 'Error: 요청 실패 (네트워크 또는 서버 연결 문제)', null);
  }
}

// --- 메시지 DOM에 추가 함수 (오류 수정) ---
function appendMessage(sender, text, imageUrl = null) {
    const currentChatLog = document.getElementById('chat-log');
    if (!currentChatLog) {
        console.error("appendMessage: chat-log element not found!");
        return;
    }

    const messageElem = document.createElement('div');
    messageElem.classList.add('message', sender);

    // ***** 수정: text를 항상 문자열로 처리 *****
    const textAsString = String(text ?? ''); // null, undefined 포함하여 문자열로 변환

    if (sender === 'user') {
        // 사용자 메시지: 문자열화된 텍스트 표시
        messageElem.textContent = textAsString; 
    } else { // sender === 'bot'
        // 1. 이미지 추가 (기존과 동일)
        if (imageUrl) {
            const contentImg = document.createElement('img');
            contentImg.classList.add('bot-big-img');
            contentImg.src = imageUrl;
            contentImg.alt = "챗봇 이미지";
            contentImg.onerror = () => {
                console.warn(`Failed to load bot image: ${imageUrl}`);
                contentImg.alt = "이미지 로드 실패";
            };
            messageElem.appendChild(contentImg);
        }

        // 2. 텍스트 추가 (수정: 문자열 확인 및 trim)
        // 비어 있지 않은 문자열인 경우에만 추가
        if (textAsString.trim() !== '') { 
            const messageTextDiv = document.createElement('div');
            messageTextDiv.classList.add('bot-text');
            messageTextDiv.textContent = textAsString; // 문자열 사용
            messageElem.appendChild(messageTextDiv);
        }
    }

    currentChatLog.appendChild(messageElem);
    // 스크롤 조정
    setTimeout(() => { currentChatLog.scrollTop = currentChatLog.scrollHeight; }, 50);
}


// --- 엔터키 또는 전송 버튼으로 전송 ---
if (userMessageInput && sendBtn) {
    userMessageInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter' && !event.isComposing) {
            sendMessage();
        }
    });
    sendBtn.addEventListener('click', sendMessage);
} else {
    console.error("Input field or send button not found.");
}


// --- 모달 열기/닫기 함수 ---
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.style.display = 'block';
  else console.error(`Modal with id ${modalId} not found.`);
}
function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.style.display = 'none';
  else console.error(`Modal with id ${modalId} not found.`);
}

// --- 미디어 버튼 이벤트 리스너 ---
if (videoBtn) {
    videoBtn.addEventListener('click', () => openModal('videoModal'));
}
if (imageBtn) {
    imageBtn.addEventListener('click', () => openModal('imageModal'));
}

// --- 모달 닫기 버튼들 이벤트 리스너 ---
document.querySelectorAll('.modal-close').forEach(btn => {
  btn.addEventListener('click', () => {
    const modalId = btn.dataset.closeModal;
    if (modalId) closeModal(modalId);
  });
});


// ==================================================
// ***** 초기 메시지 표시 (수정: 중복 appendMessage 제거) *****
// ==================================================
function displayInitialBotMessage() {
    if (chatLog && botId === '2') {
        const initialImageUrl = '/static/images/chatbot2/gallery08.png'; 
        const initialText = `안녕하세요? 손님, 은하수 식당의 월야입니다. 어떤 이야기를 나누고 싶으세요?`; 

        // --- DOM에 직접 추가하는 부분 ---
        const messageElem = document.createElement('div');
        messageElem.classList.add('message', 'bot');
        
        // 이미지 추가
        const contentImg = document.createElement('img');
        contentImg.classList.add('bot-big-img'); 
        contentImg.src = initialImageUrl;
        contentImg.alt = "은하수 식당";
        messageElem.appendChild(contentImg);
        
        // 텍스트 추가
        const messageTextDiv = document.createElement('div');
        messageTextDiv.classList.add('bot-text');
        messageTextDiv.textContent = initialText;
        messageElem.appendChild(messageTextDiv);
        
        chatLog.appendChild(messageElem);
        chatLog.scrollTop = chatLog.scrollHeight;
        // --- DOM 직접 추가 끝 ---

        // ***** 아래 중복 호출 제거 *****
        // appendMessage('bot', initialText, initialImageUrl); // << 제거

        console.log("Initial message for chatbot 2 displayed."); 
    } else {
        if (!chatLog) console.error("Initial message: chat-log element not found.");
    }
}

// --- 스크립트 로드 완료 후 초기 메시지 함수 호출 ---
if (document.readyState === 'loading') { 
    document.addEventListener('DOMContentLoaded', displayInitialBotMessage);
} else { 
    displayInitialBotMessage();
}
// ==================================================