// 파일 경로: static/js/chatbot2.js

// chat-area에서 bot_id와 bot_image_url 정보 꺼내기
const chatArea = document.querySelector('.chat-area');
const botImageUrl = chatArea.dataset.botImageUrl;
const botId = chatArea.dataset.botId; // 이 파일이 로드될 때는 botId가 '2'일 것입니다.

// 주요 DOM 요소
const chatLog = document.getElementById('chat-log');
const userMessageInput = document.getElementById('user-message');
const sendBtn = document.getElementById('send-btn');
const videoBtn = document.getElementById('videoBtn'); // null일 수 있음 (HTML 구조 확인)
const imageBtn = document.getElementById('imageBtn'); // null일 수 있음 (HTML 구조 확인)

// --- 메시지 전송 함수 ---
async function sendMessage() {
  const message = userMessageInput.value.trim();
  if (!message || !botId) {
      console.error("메시지가 없거나 botId가 유효하지 않습니다.");
      return;
  }

  appendMessage('user', message, null); // 사용자 메시지 표시 (imageUrl은 null)
  userMessageInput.value = '';
  // Optional: show loading indicator here

  try {
      const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bot_id: parseInt(botId), message: message })
      });

      // Optional: hide loading indicator here

      if (!response.ok) {
          let errorData = { error: `서버 응답 오류: ${response.status}` };
          try { errorData = await response.json(); } catch (e) {}
          console.error("Server error:", errorData);
          appendMessage('bot', `Error: ${errorData.error || response.statusText}`, null);
          return;
      }

      const data = await response.json();
      // 디버깅 로그 (필요시 유지)
      console.log("DEBUG: 서버로부터 받은 전체 데이터:", JSON.stringify(data, null, 2));
      console.log("DEBUG: data.reply 값:", data.reply);
      console.log("DEBUG: data.reply 타입:", typeof data.reply);

      // ***** 서버 응답 처리 (한 번만 수행) *****
      if (data.reply || data.image_url) {
          // reply 또는 image_url이 있으면 appendMessage 호출
          appendMessage('bot', data.reply || '', data.image_url); 
      } else if (data.error) {
          // 서버가 에러를 응답한 경우
          console.error("Server returned error:", data.error);
          appendMessage('bot', 'Error: ' + data.error, null);
      } else {
          // 예상치 못한 응답 형식의 경우
          console.error("Unexpected response format:", data);
          appendMessage('bot', 'Error: 서버로부터 예기치 않은 응답을 받았습니다.', null);
      }

  } catch (err) {
      // Optional: hide loading indicator here
      console.error("Fetch Error:", err);
      appendMessage('bot', 'Error: 요청 실패 (네트워크 또는 서버 연결 문제)', null);
  }
}

// --- 메시지 DOM에 추가 함수 ---
// ***** imageUrl 인자 추가 *****
// --- 메시지 DOM에 추가 함수 ---
// ***** imageUrl 인자 추가 및 내부 로직 정리 *****
function appendMessage(sender, text, imageUrl = null) {
  const currentChatLog = document.getElementById('chat-log');
  if (!currentChatLog) {
      console.error("appendMessage: chat-log element not found!");
      return;
  }

  const messageElem = document.createElement('div');
  messageElem.classList.add('message', sender);

  if (sender === 'user') {
      // 사용자 메시지: 텍스트만 표시
      messageElem.textContent = text;
  } else { // sender === 'bot'

      // 1. (선택적) 내용 이미지 추가
      if (imageUrl) {
          const contentImg = document.createElement('img');
          contentImg.classList.add('bot-big-img'); // 이미지 스타일용 클래스
          contentImg.src = imageUrl;
          contentImg.alt = "챗봇 이미지";
          contentImg.onerror = () => {
              console.warn(`Failed to load bot image: ${imageUrl}`);
              contentImg.alt = "이미지 로드 실패";
          };
          messageElem.appendChild(contentImg); // 이미지를 먼저 추가
      }

      // 2. (선택적) 텍스트 내용 추가 (한 번만!)
      // text가 비어있지 않은 경우에만 .bot-text div 생성 및 추가
      if (text && text.trim() !== '') {
          const messageTextDiv = document.createElement('div');
          messageTextDiv.classList.add('bot-text'); // 텍스트 스타일용 클래스
          messageTextDiv.textContent = text; // textContent 사용!
          messageElem.appendChild(messageTextDiv); // 이미지 뒤 또는 단독으로 텍스트 추가
      }

      // 3. 아바타 처리 (주석 처리 또는 삭제)
      // 이전 코드에 있던 중복된 아바타 + 텍스트 추가 로직은 여기서 삭제되었습니다.
      // 만약 아바타가 필요하다면 CSS로 처리하거나 다른 방식으로 추가해야 합니다.
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
// videoBtn, imageBtn 요소가 존재하는지 확인 후 이벤트 리스너 추가
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
// ***** 추가된 코드: 초기 메시지 표시 *****
// ==================================================
function displayInitialBotMessage() {
  // 이 스크립트(chatbot2.js)는 botId가 2일 때만 로드되므로,
  // botId 체크는 사실상 불필요하지만 명시적으로 남겨둘 수 있습니다.
  if (chatLog && botId === '2') {
    const userName = chatArea.dataset.username || "손";
    const initialImageUrl = '/static/images/chatbot2/gallery18.png'; // Flask static 폴더 기준 경로
    const initialText = `안녕하세요? ${userName}님, 심야식당입니다`;

    // 메시지 요소 생성
    const messageElem = document.createElement('div');
    messageElem.classList.add('message', 'bot');

    // 내용 이미지(gallery18) 요소 생성 및 추가
    const contentImg = document.createElement('img');
    contentImg.classList.add('bot-big-img'); // 기존 클래스 사용
    contentImg.src = initialImageUrl;
    contentImg.alt = "심야식당";
    // .bot-big-img 스타일에 따라 표시됨 (max-width: 300px, margin: 0 auto 8px 등)
    messageElem.appendChild(contentImg);

    // 텍스트 메시지 요소 생성 및 추가
    const messageTextDiv = document.createElement('div');
    messageTextDiv.classList.add('bot-text');
    messageTextDiv.textContent = initialText;
    messageElem.appendChild(messageTextDiv);

    // 완성된 메시지 요소를 chatLog에 추가
    chatLog.appendChild(messageElem);
    // 스크롤을 맨 아래로 이동
    chatLog.scrollTop = chatLog.scrollHeight;

    console.log("Initial message for chatbot 2 displayed."); // 확인용 로그
  } else {
    if (!chatLog) console.error("Initial message: chat-log element not found.");
    // botId가 2가 아닌 경우는 이 파일이 로드되지 않아야 함
  }
}

// --- 스크립트 로드 완료 후 초기 메시지 함수 호출 ---
// DOMContentLoaded는 HTML 파싱 완료 시점, window.onload는 모든 리소스(이미지 등) 로드 완료 시점
// 여기서는 DOM 요소 접근만 필요하므로 DOMContentLoaded가 일반적으로 더 빠름
// 하지만 이 스크립트 자체가 동적으로 로드되므로, 그냥 바로 호출해도 문제는 없을 가능성이 높음.
// 안전하게 하려면 DOMContentLoaded 사용
if (document.readyState === 'loading') { // 아직 로딩 중이면
    document.addEventListener('DOMContentLoaded', displayInitialBotMessage);
} else { // 이미 로딩 완료되었으면
    displayInitialBotMessage();
}
// ==================================================
// ***** 추가된 코드 끝 *****
// ==================================================