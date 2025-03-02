// DOM Elements
const userInput = document.getElementById('userInput');
const translationOutput = document.getElementById('translationOutput');
const micBtn = document.getElementById('micBtn');
const clearInput = document.getElementById('clearInput');
const copyBtn = document.getElementById('copyBtn');
const speakBtn = document.getElementById('speakBtn');
const chatHistory = document.getElementById('chatHistory');
const recordingStatus = document.getElementById('recordingStatus');
const typingIndicator = document.getElementById('typingIndicator');

// Initialize speech recognition
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
}

// Initialize speech synthesis
const synthesis = window.speechSynthesis;

// State
let isRecording = false;

// Show/hide typing indicator
function toggleTypingIndicator(show) {
    typingIndicator.classList.toggle('hidden', !show);
}

// Add message to chat history
function addToHistory(original, translation) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bg-white p-4 rounded-lg shadow';
    
    const originalText = document.createElement('p');
    originalText.className = 'text-gray-700 mb-2';
    originalText.textContent = original;
    
    const translatedText = document.createElement('p');
    translatedText.className = 'text-blue-600 cursor-pointer';
    translatedText.textContent = translation;
    translatedText.onclick = () => speak(translation);
    
    messageDiv.appendChild(originalText);
    messageDiv.appendChild(translatedText);
    chatHistory.insertBefore(messageDiv, chatHistory.firstChild);
}

// Speech synthesis
function speak(text, lang = 'fr-FR') {
    if (synthesis.speaking) {
        synthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.9;
    synthesis.speak(utterance);
}

// Copy text to clipboard
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        // Show quick feedback
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => {
            copyBtn.innerHTML = originalText;
        }, 2000);
    } catch (err) {
        console.error('Failed to copy text:', err);
    }
}

// Handle translation
async function handleTranslation(text) {
    if (!text.trim()) return;

    // Show typing indicator
    toggleTypingIndicator(true);
    translationOutput.textContent = '';

    try {
        const response = await fetch('/api/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error('Translation failed');
        }

        const data = await response.json();
        
        // Hide typing indicator
        toggleTypingIndicator(false);

        // Display translation
        translationOutput.textContent = data.translation;
        
        // Add to history
        addToHistory(text, data.translation);

    } catch (error) {
        console.error('Error:', error);
        toggleTypingIndicator(false);
        translationOutput.textContent = 'An error occurred. Please try again.';
    }
}

// Event Listeners
userInput.addEventListener('input', () => {
    const text = userInput.value.trim();
    if (text) {
        handleTranslation(text);
    } else {
        translationOutput.textContent = '';
    }
});

clearInput.addEventListener('click', () => {
    userInput.value = '';
    translationOutput.textContent = '';
    userInput.focus();
});

copyBtn.addEventListener('click', () => {
    const text = translationOutput.textContent;
    if (text) {
        copyToClipboard(text);
    }
});

speakBtn.addEventListener('click', () => {
    const text = translationOutput.textContent;
    if (text) {
        speak(text);
    }
});

// Speech Recognition
if (recognition) {
    recognition.onstart = () => {
        isRecording = true;
        recordingStatus.classList.remove('hidden');
        micBtn.classList.add('text-red-500');
    };

    recognition.onend = () => {
        isRecording = false;
        recordingStatus.classList.add('hidden');
        micBtn.classList.remove('text-red-500');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        handleTranslation(transcript);
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        isRecording = false;
        recordingStatus.classList.add('hidden');
        micBtn.classList.remove('text-red-500');
    };
}

micBtn.addEventListener('click', () => {
    if (!recognition) {
        alert('Speech recognition is not supported in your browser.');
        return;
    }

    if (isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isRecording && recognition) {
        recognition.stop();
    }
});