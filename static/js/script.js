// -------- Send Message --------
function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();

    if (!message) return;

    // Add user bubble
    addMessage(message, 'user');
    input.value = '';

    // Disable send button
    document.getElementById('sendBtn').disabled = true;

    // Show typing indicator
    showTyping();

    // Call Flask API
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        removeTyping();
        addMessage(data.response, 'bot');
        document.getElementById('sendBtn').disabled = false;
    })
    .catch(err => {
        removeTyping();
        addMessage('Sorry, something went wrong. Please try again!', 'bot');
        document.getElementById('sendBtn').disabled = false;
    });
}

// -------- Add Message Bubble --------
function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'bot' ? 'bot-message' : 'user-message');

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = sender === 'bot' ? '🎓' : '👤';

    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.innerHTML = text.replace(/\n/g, '<br>');

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// -------- Typing Indicator --------
function showTyping() {
    const chatMessages = document.getElementById('chatMessages');

    const typingDiv = document.createElement('div');
    typingDiv.classList.add('message', 'bot-message');
    typingDiv.id = 'typingIndicator';

    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = '🎓';

    const typing = document.createElement('div');
    typing.classList.add('typing-indicator');
    typing.innerHTML = '<span></span><span></span><span></span>';

    typingDiv.appendChild(avatar);
    typingDiv.appendChild(typing);

    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

// -------- Suggestion Buttons --------
function sendSuggestion(text) {
    document.getElementById('userInput').value = text;
    sendMessage();
}

// -------- Enter Key Support --------
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') sendMessage();
});