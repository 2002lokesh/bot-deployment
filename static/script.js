const form = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const chatContainer = document.getElementById('chat-container');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const userMessage = userInput.value;
    addMessage(userMessage, 'user');
    const response = await sendMessage(userMessage);
    addMessage(response.response, 'chatbot');
    userInput.value = '';
});

async function sendMessage(userMessage) {
    const response = await fetch('/chatbot/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `user_input=${encodeURIComponent(userMessage)}`,
    });
    return await response.json();
}

function addMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add(sender === 'user' ? 'user-message' : 'chatbot-message');
    messageElement.textContent = message;
    chatContainer.appendChild(messageElement);
}
