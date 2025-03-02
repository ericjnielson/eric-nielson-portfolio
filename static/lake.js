// Initialize Socket.IO with more robust error handling and reconnection
const socket = io({
    path: '/ws/socket.io',
    transports: ['polling', 'websocket'],
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    timeout: 20000,
    autoConnect: true
});

// Connection state tracking
let socketConnected = false;

// DOM Elements
const startButton = document.getElementById('startTraining');
const stopButton = document.getElementById('stopButton');
const saveButton = document.getElementById('saveModel');
const loadButton = document.getElementById('loadModel');
const randomizeButton = document.getElementById('randomizeMap');
const mapSizeSelect = document.getElementById('mapSize');
const learningRateSlider = document.getElementById('learningRate');
const discountFactorSlider = document.getElementById('discountFactor');
const explorationRateSlider = document.getElementById('explorationRate');
const learningRateValue = document.getElementById('learningRateValue');
const discountFactorValue = document.getElementById('discountFactorValue');
const explorationRateValue = document.getElementById('explorationRateValue');
const modelStatus = document.getElementById('modelStatus');
const trainingProgress = document.getElementById('trainingProgress');
const environmentVisualization = document.getElementById('environmentVisualization');
const avgReward = document.getElementById('avgReward');
const episodeCount = document.getElementById('episodeCount');
const successRate = document.getElementById('successRate');

// Training state
let isTraining = false;

// Update slider value displays
function updateSliderDisplay(slider, display) {
    display.textContent = Number(slider.value).toFixed(
        slider.id === 'learningRate' ? 4 : 3
    );
}

// Initialize slider displays
updateSliderDisplay(learningRateSlider, learningRateValue);
updateSliderDisplay(discountFactorSlider, discountFactorValue);
updateSliderDisplay(explorationRateSlider, explorationRateValue);

// Add input event listeners
learningRateSlider.addEventListener('input', () => updateSliderDisplay(learningRateSlider, learningRateValue));
discountFactorSlider.addEventListener('input', () => updateSliderDisplay(discountFactorSlider, discountFactorValue));
explorationRateSlider.addEventListener('input', () => updateSliderDisplay(explorationRateSlider, explorationRateValue));

// Socket event handlers with improved error handling
socket.on('connect', () => {
    console.log('Socket connected successfully');
    console.log('Transport type:', socket.io.engine.transport.name);
    modelStatus.textContent = 'Connected';
    modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
    socketConnected = true;
    
    // Request initial state update when connected
    updateCurrentState();
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
    modelStatus.textContent = 'Connection Error';
    modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
    socketConnected = false;
    
    // Alert the user about connection issues
    showNotification('Connection error: Unable to establish WebSocket connection. Trying to reconnect...', true);
});

socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
    modelStatus.textContent = 'Disconnected';
    modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800';
    socketConnected = false;
    
    // If training is in progress, try to stop it through regular HTTP
    if (isTraining) {
        stopTraining();
    }
    
    // Try to reconnect manually if socket.io doesn't
    setTimeout(() => {
        if (!socketConnected) {
            console.log('Attempting manual reconnection...');
            socket.connect();
        }
    }, 3000);
});

socket.on('training_update', (data) => {
    updateMetrics(data.metrics);
    if (data.frame) {
        updateVisualization(data.frame);
    }
});

socket.on('training_complete', (data) => {
    console.log('Training complete:', data);
    stopTraining();
    updateMetrics(data.metrics);
    if (data.frame) {
        updateVisualization(data.frame);
    }
    showNotification('Training complete! ' + 
        `Success rate: ${data.metrics.success_rate.toFixed(1)}%, ` +
        `Episodes: ${data.metrics.episode_count}`);
});

socket.on('training_error', (data) => {
    showNotification(data.error, true);
    stopTraining();
});

// Fallback HTTP communication for environments where WebSockets fail
async function updateStateViaHTTP() {
    if (!socketConnected) {
        try {
            await updateCurrentState();
        } catch (error) {
            console.error("Error updating state via HTTP:", error);
        }
    }
}

// Set up periodic state updates (every 5 seconds) as a fallback
setInterval(updateStateViaHTTP, 5000);

// Training control functions with improved error handling
async function startTraining() {
    try {
        console.log("Starting training...");
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                learning_rate: parseFloat(learningRateSlider.value),
                discount_factor: parseFloat(discountFactorSlider.value),
                exploration_rate: parseFloat(explorationRateSlider.value)
            }),
            timeout: 10000 // 10 second timeout
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to start training');
        }

        isTraining = true;
        startButton.disabled = true;
        stopButton.disabled = false;
        randomizeButton.disabled = true;
        mapSizeSelect.disabled = true;
        modelStatus.textContent = 'Training';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800';

    } catch (error) {
        console.error("Start training error:", error);
        showNotification('Error starting training: ' + error.message, true);
    }
}

async function stopTraining() {
    try {
        console.log("Stopping training...");
        const response = await fetch('/stop_training', {
            method: 'POST',
            timeout: 10000 // 10 second timeout
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to stop training');
        }

        isTraining = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        randomizeButton.disabled = false;
        mapSizeSelect.disabled = false;
        modelStatus.textContent = 'Model Ready';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';

        // Get final state update
        updateCurrentState();

    } catch (error) {
        console.error("Stop training error:", error);
        showNotification('Error stopping training: ' + error.message, true);
        
        // Reset UI even on error to allow restart
        isTraining = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        randomizeButton.disabled = false;
        mapSizeSelect.disabled = false;
    }
}

async function randomizeMap() {
    try {
        console.log("Randomizing map...");
        randomizeButton.disabled = true;
        
        const response = await fetch('/randomize_map', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                size: parseInt(mapSizeSelect.value)
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to randomize map');
        }

        // If we got frame data directly, use it
        if (data.frame) {
            updateVisualization(data.frame);
            if (data.metrics) {
                updateMetrics(data.metrics);
            }
        } else {
            // Otherwise get current state
            await updateCurrentState();
        }
        
        showNotification('Map randomized successfully');

    } catch (error) {
        console.error("Randomize map error:", error);
        showNotification('Error randomizing map: ' + error.message, true);
    } finally {
        randomizeButton.disabled = false;
    }
}

async function saveModelState() {
    try {
        saveButton.disabled = true;
        
        const response = await fetch('/save_model', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to save model');
        }
        
        showNotification('Model saved successfully');
    } catch (error) {
        showNotification('Error saving model: ' + error.message, true);
    } finally {
        saveButton.disabled = false;
    }
}

async function loadModelState() {
    try {
        loadButton.disabled = true;
        
        const response = await fetch('/load_model', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to load model');
        }
        
        await updateCurrentState();
        showNotification('Model loaded successfully');
    } catch (error) {
        showNotification('Error loading model: ' + error.message, true);
    } finally {
        loadButton.disabled = false;
    }
}

// Update functions
function updateMetrics(metrics) {
    if (!metrics) return;
    
    avgReward.textContent = (metrics.average_reward || 0).toFixed(2);
    episodeCount.textContent = metrics.episode_count || 0;
    successRate.textContent = `${(metrics.success_rate || 0).toFixed(1)}%`;
    
    // Update progress bar (assuming 1000 episodes as goal)
    const progress = (metrics.episode_count / 1000) * 100;
    trainingProgress.style.width = `${Math.min(progress, 100)}%`;
}

function updateVisualization(frameData) {
    if (!frameData) {
        console.warn("No frame data provided");
        return;
    }

    try {
        // Clear previous visualization
        while (environmentVisualization.firstChild) {
            environmentVisualization.removeChild(environmentVisualization.firstChild);
        }

        // Create and add new visualization
        const img = document.createElement('img');
        img.onload = () => {
            console.log(`Image loaded successfully: ${img.width}x${img.height}`);
        };
        
        img.onerror = (e) => {
            console.error("Error loading image:", e);
        };
        
        img.src = `data:image/jpeg;base64,${frameData}`;
        img.className = 'w-full h-full object-contain';
        environmentVisualization.appendChild(img);
    } catch (error) {
        console.error("Error in updateVisualization:", error);
    }
}

async function updateCurrentState() {
    try {
        const response = await fetch('/get_state');
        if (!response.ok) {
            throw new Error('Failed to get current state');
        }
        
        const data = await response.json();
        updateMetrics(data.metrics);
        if (data.frame) {
            console.log('Got frame from state update, updating visualization');
            updateVisualization(data.frame);
        }
    } catch (error) {
        console.error('Error getting current state:', error);
        showNotification('Error getting current state: ' + error.message, true);
    }
}

// Notification function
function showNotification(message, isError = false) {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg text-white ${
        isError ? 'bg-red-600' : 'bg-green-600'
    } shadow-lg z-50 transition-opacity duration-300`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Fade out and remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Event Listeners
startButton.addEventListener('click', startTraining);
stopButton.addEventListener('click', stopTraining);
saveButton.addEventListener('click', saveModelState);
loadButton.addEventListener('click', loadModelState);
randomizeButton.addEventListener('click', randomizeMap);
mapSizeSelect.addEventListener('change', randomizeMap);

// Initialize
stopButton.disabled = true;
console.log('Initializing RL training interface');
updateCurrentState();

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isTraining) {
        console.log('Page hidden, stopping training');
        stopTraining();
    } else if (!document.hidden) {
        console.log('Page visible, updating state');
        updateCurrentState();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isTraining) {
        console.log('Page unloading, stopping training');
        navigator.sendBeacon('/stop_training', '{}');
        isTraining = false;
    }
});