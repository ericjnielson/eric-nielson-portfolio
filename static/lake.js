// Initialize Socket.IO with optimizations for Cloud Run
const socket = io({
    path: '/ws/socket.io',
    transports: ['websocket'], // Force WebSockets only to avoid polling issues
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 3000,
    reconnectionDelayMax: 10000,
    timeout: 20000,
    forceNew: true
});

// Connection state and request management
let socketConnected = false;
let requestInProgress = false;
let lastRequestTime = 0;
const REQUEST_COOLDOWN = 2000; // Increased cooldown for Cloud Run
let pendingStateUpdate = false;

// DOM Elements
const startButton = document.getElementById('startTraining');
const stopButton = document.getElementById('stopTraining');
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
    if (!slider || !display) return;
    display.textContent = Number(slider.value).toFixed(
        slider.id === 'learningRate' ? 4 : 3
    );
}

// Initialize slider displays
if (learningRateSlider && learningRateValue) {
    updateSliderDisplay(learningRateSlider, learningRateValue);
}
if (discountFactorSlider && discountFactorValue) {
    updateSliderDisplay(discountFactorSlider, discountFactorValue);
}
if (explorationRateSlider && explorationRateValue) {
    updateSliderDisplay(explorationRateSlider, explorationRateValue);
}

// Add input event listeners
if (learningRateSlider && learningRateValue) {
    learningRateSlider.addEventListener('input', () => updateSliderDisplay(learningRateSlider, learningRateValue));
}
if (discountFactorSlider && discountFactorValue) {
    discountFactorSlider.addEventListener('input', () => updateSliderDisplay(discountFactorSlider, discountFactorValue));
}
if (explorationRateSlider && explorationRateValue) {
    explorationRateSlider.addEventListener('input', () => updateSliderDisplay(explorationRateSlider, explorationRateValue));
}

// Socket event handlers
socket.on('connect', () => {
    console.log('Socket connected successfully');
    console.log('Transport type:', socket.io.engine.transport.name);
    if (modelStatus) {
        modelStatus.textContent = 'Connected';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
    }
    socketConnected = true;
    
    // Request initial state update when connected, with delay to ensure connection stability
    setTimeout(() => updateCurrentState(), 1000);
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
    if (modelStatus) {
        modelStatus.textContent = 'Connection Error';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
    }
    socketConnected = false;
    
    // Attempt to switch to alternate transport if connection fails
    if (socket.io.engine && socket.io.engine.transport) {
        console.log('Trying to use alternate transport');
        socket.io.engine.transport.on('upgrade', (transport) => {
            console.log(`Transport upgraded to ${transport.name}`);
        });
    }
    
    showNotification('Connection error. The server may be experiencing high load.', true);
});

socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
    if (modelStatus) {
        modelStatus.textContent = 'Disconnected';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800';
    }
    socketConnected = false;
    
    // If training is in progress, try to stop it
    if (isTraining) {
        stopTraining(true); // Force stop without making request
    }
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
    showNotification(data.error || 'An error occurred during training', true);
    stopTraining();
});

// Rate limiting helper
function canMakeRequest() {
    const now = Date.now();
    if (requestInProgress || (now - lastRequestTime < REQUEST_COOLDOWN)) {
        console.log("Request throttled, please wait a moment");
        return false;
    }
    return true;
}

// Training control functions
async function startTraining() {
    if (!socketConnected) {
        showNotification("Cannot start training: not connected to server", true);
        return;
    }
    
    if (isTraining) {
        console.log("Training already in progress");
        return;
    }
    
    if (!canMakeRequest()) {
        showNotification("Please wait a moment before trying again", true);
        return;
    }
    
    try {
        console.log("Starting training...");
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (startButton) startButton.disabled = true;
        
        // Add request ID to make request unique and avoid caching
        const requestId = Date.now().toString();
        const response = await fetch(`/start_training?rid=${requestId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({
                learning_rate: parseFloat(learningRateSlider?.value || 0.001),
                discount_factor: parseFloat(discountFactorSlider?.value || 0.99),
                exploration_rate: parseFloat(explorationRateSlider?.value || 0.1)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        await response.json();
        
        isTraining = true;
        if (startButton) startButton.disabled = true;
        if (stopButton) stopButton.disabled = false;
        if (randomizeButton) randomizeButton.disabled = true;
        if (mapSizeSelect) mapSizeSelect.disabled = true;
        
        if (modelStatus) {
            modelStatus.textContent = 'Training';
            modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800';
        }
        
    } catch (error) {
        console.error("Start training error:", error);
        showNotification('Error starting training: ' + error.message, true);
        if (startButton) startButton.disabled = false;
    } finally {
        requestInProgress = false;
    }
}

async function stopTraining(force = false) {
    if (!isTraining && !force) {
        console.log("No training in progress");
        return;
    }
    
    if (!force && !canMakeRequest()) {
        return;
    }
    
    try {
        console.log("Stopping training...");
        
        if (!force) {
            requestInProgress = true;
            lastRequestTime = Date.now();
        }
        
        if (stopButton) stopButton.disabled = true;
        
        // Only make the request if not forced
        if (!force && socketConnected) {
            try {
                // Add request ID
                const requestId = Date.now().toString();
                const response = await fetch(`/stop_training?rid=${requestId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Cache-Control': 'no-cache',
                        'Pragma': 'no-cache'
                    },
                    body: JSON.stringify({})
                });
                
                if (!response.ok) {
                    console.warn(`Stop training response not OK: ${response.status}`);
                }
            } catch (error) {
                console.warn("Error during stop request:", error);
                // Continue with UI updates even if request fails
            }
        }
    } finally {
        // Always update UI state
        isTraining = false;
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (randomizeButton) randomizeButton.disabled = false;
        if (mapSizeSelect) mapSizeSelect.disabled = false;
        
        if (modelStatus) {
            modelStatus.textContent = 'Model Ready';
            modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
        }
        
        if (!force) {
            requestInProgress = false;
            // Update current state after a delay
            setTimeout(() => updateCurrentState(), 1000);
        }
    }
}

async function randomizeMap() {
    if (!socketConnected) {
        showNotification("Cannot randomize map: not connected to server", true);
        return;
    }
    
    if (isTraining) {
        showNotification("Cannot randomize map while training", true);
        return;
    }
    
    if (!canMakeRequest()) {
        showNotification("Please wait a moment before trying again", true);
        return;
    }
    
    try {
        console.log("Randomizing map...");
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (randomizeButton) randomizeButton.disabled = true;
        
        // Add request ID
        const requestId = Date.now().toString();
        const response = await fetch(`/randomize_map?rid=${requestId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({
                size: parseInt(mapSizeSelect?.value || 8)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
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
        requestInProgress = false;
        if (randomizeButton) randomizeButton.disabled = false;
    }
}

async function saveModelState() {
    if (!socketConnected) {
        showNotification("Cannot save model: not connected to server", true);
        return;
    }
    
    if (isTraining) {
        showNotification("Cannot save model while training", true);
        return;
    }
    
    if (!canMakeRequest()) {
        showNotification("Please wait a moment before trying again", true);
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (saveButton) saveButton.disabled = true;
        
        // Add request ID
        const requestId = Date.now().toString();
        const response = await fetch(`/save_model?rid=${requestId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({})
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        await response.json();
        showNotification("Model saved successfully");
        
    } catch (error) {
        console.error("Save model error:", error);
        showNotification('Error saving model: ' + error.message, true);
    } finally {
        requestInProgress = false;
        if (saveButton) saveButton.disabled = false;
    }
}

async function loadModelState() {
    if (!socketConnected) {
        showNotification("Cannot load model: not connected to server", true);
        return;
    }
    
    if (isTraining) {
        showNotification("Cannot load model while training", true);
        return;
    }
    
    if (!canMakeRequest()) {
        showNotification("Please wait a moment before trying again", true);
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (loadButton) loadButton.disabled = true;
        
        // Add request ID
        const requestId = Date.now().toString();
        const response = await fetch(`/load_model?rid=${requestId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            body: JSON.stringify({})
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        await response.json();
        
        // Update state after loading
        setTimeout(() => updateCurrentState(), 1000);
        showNotification("Model loaded successfully");
        
    } catch (error) {
        console.error("Load model error:", error);
        showNotification('Error loading model: ' + error.message, true);
    } finally {
        requestInProgress = false;
        if (loadButton) loadButton.disabled = false;
    }
}

// Update functions
function updateMetrics(metrics) {
    if (!metrics) return;
    
    if (avgReward) avgReward.textContent = (metrics.average_reward || 0).toFixed(2);
    if (episodeCount) episodeCount.textContent = metrics.episode_count || 0;
    if (successRate) successRate.textContent = `${(metrics.success_rate || 0).toFixed(1)}%`;
    
    // Update progress bar (assuming 1000 episodes as goal)
    if (trainingProgress) {
        const progress = (metrics.episode_count / (metrics.max_episodes || 500)) * 100;
        trainingProgress.style.width = `${Math.min(progress, 100)}%`;
    }
}

function updateVisualization(frameData) {
    if (!frameData || !environmentVisualization) {
        return;
    }

    try {
        // Clear previous visualization
        while (environmentVisualization.firstChild) {
            environmentVisualization.removeChild(environmentVisualization.firstChild);
        }

        // Create and add new visualization
        const img = new Image();
        
        img.onload = () => {
            console.log(`Image loaded successfully: ${img.width}x${img.height}`);
        };
        
        img.onerror = (e) => {
            console.error("Error loading image:", e);
            // Show fallback visualization
            showFallbackVisualization();
        };
        
        img.src = `data:image/jpeg;base64,${frameData}`;
        img.className = 'w-full h-full object-contain';
        environmentVisualization.appendChild(img);
    } catch (error) {
        console.error("Error updating visualization:", error);
        showFallbackVisualization();
    }
}

function showFallbackVisualization() {
    if (!environmentVisualization) return;
    
    const fallbackDiv = document.createElement('div');
    fallbackDiv.textContent = "Visualization unavailable";
    fallbackDiv.style.width = "100%";
    fallbackDiv.style.height = "100%";
    fallbackDiv.style.display = "flex";
    fallbackDiv.style.alignItems = "center";
    fallbackDiv.style.justifyContent = "center";
    fallbackDiv.style.backgroundColor = "#f0f0f0";
    fallbackDiv.style.color = "#666";
    
    // Clear and add fallback
    while (environmentVisualization.firstChild) {
        environmentVisualization.removeChild(environmentVisualization.firstChild);
    }
    environmentVisualization.appendChild(fallbackDiv);
}

// Modified updateCurrentState with debouncing and request deduplication
async function updateCurrentState() {
    if (!socketConnected) {
        console.log("Cannot update state: not connected to server");
        return;
    }
    
    if (requestInProgress) {
        console.log("Request already in progress, marking for update when current request completes");
        pendingStateUpdate = true;
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        pendingStateUpdate = false;
        
        // Add request ID to prevent caching issues
        const requestId = Date.now().toString();
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout
        
        const response = await fetch(`/get_state?rid=${requestId}`, {
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            },
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`Failed to get state: ${response.status}`);
        }
        
        const data = await response.json();
        
        updateMetrics(data.metrics);
        if (data.frame) {
            updateVisualization(data.frame);
        }
        
    } catch (error) {
        console.error('Error getting current state:', error);
        if (error.name === 'AbortError') {
            console.log('State update request timed out');
        } else {
            showNotification('Failed to get current state: ' + error.message, true);
        }
    } finally {
        requestInProgress = false;
        
        // If there's a pending update request, process it after a short delay
        if (pendingStateUpdate) {
            setTimeout(() => {
                pendingStateUpdate = false;
                updateCurrentState();
            }, 1000);
        }
    }
}

// Notification function
function showNotification(message, isError = false) {
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification-message');
    existingNotifications.forEach(notification => {
        notification.remove();
    });
    
    const notification = document.createElement('div');
    notification.className = `notification-message fixed top-4 right-4 px-6 py-3 rounded-lg text-white ${
        isError ? 'bg-red-600' : 'bg-green-600'
    } shadow-lg z-50 transition-opacity duration-300 opacity-0`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Fade in
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 10);
    
    // Fade out and remove after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, isError ? 5000 : 3000);
}

// Event Listeners
if (startButton) {
    startButton.addEventListener('click', startTraining);
}
if (stopButton) {
    stopButton.addEventListener('click', () => stopTraining());
}
if (saveButton) {
    saveButton.addEventListener('click', saveModelState);
}
if (loadButton) {
    loadButton.addEventListener('click', loadModelState);
}
if (randomizeButton) {
    randomizeButton.addEventListener('click', randomizeMap);
}
if (mapSizeSelect) {
    // Don't trigger randomize on change - let user decide
    mapSizeSelect.addEventListener('change', () => {
        console.log(`Map size selected: ${mapSizeSelect.value}x${mapSizeSelect.value}`);
    });
}

// Initialize
if (stopButton) {
    stopButton.disabled = true;
}
console.log('Initializing RL training interface');

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (isTraining) {
            console.log('Page hidden, stopping training');
            stopTraining(true);
        }
    } else {
        console.log('Page visible, updating state');
        if (socketConnected) {
            setTimeout(() => updateCurrentState(), 1000);
        } else {
            // Try to reconnect
            socket.connect();
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isTraining) {
        console.log('Page unloading, stopping training');
        // Use a sync request for unload
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/stop_training', false);  // false makes it synchronous
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({}));
        isTraining = false;
    }
});