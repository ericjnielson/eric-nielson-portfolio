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
let retryCount = 0;
const MAX_RETRIES = 5;
let requestsInProgress = 0; // Track ongoing requests
let lastRequestTime = 0; // Track when the last request was made
const MIN_REQUEST_INTERVAL = 2000; // Minimum time between requests (ms)
let initialStateLoaded = false; // Track if initial state has been loaded

// Update slider value displays
function updateSliderDisplay(slider, display) {
    if (!slider || !display) return; // Guard against null elements
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

// Socket event handlers with improved error handling
socket.on('connect', () => {
    console.log('Socket connected successfully');
    console.log('Transport type:', socket.io.engine.transport.name);
    if (modelStatus) {
        modelStatus.textContent = 'Connected';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
    }
    socketConnected = true;
    retryCount = 0;
    
    // Request initial state update when connected, but only if we haven't already loaded the state
    if (!initialStateLoaded) {
        setTimeout(() => updateCurrentState(), 500);
    }
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
    if (modelStatus) {
        modelStatus.textContent = 'Connection Error';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
    }
    socketConnected = false;
    
    // Alert the user about connection issues if retries exhausted
    if (retryCount >= MAX_RETRIES) {
        showNotification('Connection error: Unable to establish WebSocket connection. Please check your network and reload the page.', true);
    }
});

socket.on('disconnect', (reason) => {
    console.log('Socket disconnected:', reason);
    if (modelStatus) {
        modelStatus.textContent = 'Disconnected';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-yellow-100 text-yellow-800';
    }
    socketConnected = false;
    
    // If training is in progress, try to stop it through regular HTTP
    if (isTraining) {
        stopTraining();
    }
    
    // Try to reconnect manually if socket.io doesn't
    if (retryCount < MAX_RETRIES) {
        retryCount++;
        setTimeout(() => {
            if (!socketConnected) {
                console.log(`Attempting manual reconnection (attempt ${retryCount})...`);
                socket.connect();
            }
        }, 2000 * retryCount); // Exponential backoff
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
    showNotification(data.error || 'Unknown training error occurred', true);
    stopTraining();
});

socket.on('error', (error) => {
    console.error('Socket error:', error);
    showNotification('Socket error: ' + (error.message || 'Unknown error'), true);
});

// Simple fetch with shorter timeout
async function simpleFetch(url, options = {}, timeout = 5000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    options.signal = controller.signal;
    
    try {
        requestsInProgress++;
        const response = await fetch(url, options);
        clearTimeout(timeoutId);
        return response;
    } catch (error) {
        throw error;
    } finally {
        requestsInProgress--;
        clearTimeout(timeoutId);
    }
}

// Throttle function to prevent too many requests
function shouldThrottleRequest() {
    const now = Date.now();
    const timeSinceLastRequest = now - lastRequestTime;
    return timeSinceLastRequest < MIN_REQUEST_INTERVAL;
}

// Training control functions with improved error handling
async function startTraining() {
    try {
        if (isTraining || requestsInProgress > 0) {
            console.log("Training already in progress or requests pending, ignoring start request");
            return;
        }
        
        if (shouldThrottleRequest()) {
            console.log("Request throttled, please try again in a moment");
            return;
        }
        
        lastRequestTime = Date.now();
        console.log("Starting training...");
        if (startButton) startButton.disabled = true;
        
        const body = JSON.stringify({
            learning_rate: parseFloat(learningRateSlider?.value || 0.001),
            discount_factor: parseFloat(discountFactorSlider?.value || 0.99),
            exploration_rate: parseFloat(explorationRateSlider?.value || 0.1)
        });

        try {
            requestsInProgress++;
            const response = await fetch('/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: body
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to start training (${response.status}): ${errorText}`);
            }

            const data = await response.json();
            
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
            throw error;
        } finally {
            requestsInProgress--;
        }

    } catch (error) {
        console.error("Start training error:", error);
        showNotification('Error starting training: ' + error.message, true);
        if (startButton) startButton.disabled = false;
    }
}

async function stopTraining() {
    try {
        if (shouldThrottleRequest()) {
            console.log("Request throttled, please try again in a moment");
            return;
        }
        
        lastRequestTime = Date.now();
        console.log("Stopping training...");
        if (stopButton) stopButton.disabled = true;
        
        try {
            requestsInProgress++;
            const response = await fetch('/stop_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to stop training (${response.status}): ${errorText}`);
            }

            await response.json();
        } catch (error) {
            console.warn("Error stopping training, but continuing UI reset:", error);
        } finally {
            requestsInProgress--;
        }
        
        // Always reset UI state regardless of response
        isTraining = false;
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (randomizeButton) randomizeButton.disabled = false;
        if (mapSizeSelect) mapSizeSelect.disabled = false;
        
        if (modelStatus) {
            modelStatus.textContent = 'Model Ready';
            modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
        }

        // Get final state update
        setTimeout(() => {
            updateCurrentState();
        }, 1000);

    } catch (error) {
        console.error("Stop training error:", error);
        showNotification('Error stopping training: ' + error.message, true);
        
        // Reset UI even on error to allow restart
        isTraining = false;
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        if (randomizeButton) randomizeButton.disabled = false;
        if (mapSizeSelect) mapSizeSelect.disabled = false;
    }
}

async function randomizeMap() {
    if (isTraining) {
        showNotification("Cannot randomize map while training is in progress", true);
        return;
    }
    
    if (requestsInProgress > 0) {
        showNotification("Please wait for current operation to complete", true);
        return;
    }
    
    if (shouldThrottleRequest()) {
        showNotification("Please wait a moment before trying again", true);
        return;
    }
    
    lastRequestTime = Date.now();
    console.log("Randomizing map...");
    if (randomizeButton) randomizeButton.disabled = true;
    
    try {
        // Get the selected map size safely
        let mapSize = 8; // Default size
        if (mapSizeSelect && mapSizeSelect.value) {
            const sizeValue = parseInt(mapSizeSelect.value);
            if (!isNaN(sizeValue) && sizeValue > 0) {
                mapSize = sizeValue;
            }
        }
        
        requestsInProgress++;
        const response = await fetch('/randomize_map', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                size: mapSize
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to randomize map (${response.status}): ${errorText}`);
        }

        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            console.error("Error parsing randomize response:", jsonError);
            throw new Error("Received invalid response from server");
        }
        
        // Update UI with received data
        if (data && data.frame) {
            updateVisualization(data.frame);
            if (data.metrics) {
                updateMetrics(data.metrics);
            }
            showNotification('Map randomized successfully');
        } else {
            // If we didn't get frame data, try to update the state separately
            await updateCurrentState();
            showNotification('Map randomized, updating state...');
        }
    } catch (error) {
        console.error("Randomize map error:", error);
        showNotification('Error randomizing map: ' + error.message, true);
    } finally {
        requestsInProgress--;
        if (randomizeButton) randomizeButton.disabled = false;
    }
}

async function saveModelState() {
    if (isTraining || requestsInProgress > 0 || shouldThrottleRequest()) {
        showNotification("Cannot save model right now, please try again in a moment", true);
        return;
    }
    
    lastRequestTime = Date.now();
    if (saveButton) saveButton.disabled = true;
    
    try {
        requestsInProgress++;
        const response = await fetch('/save_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to save model (${response.status}): ${errorText}`);
        }

        await response.json();
        showNotification('Model saved successfully');
    } catch (error) {
        console.error("Save model error:", error);
        showNotification('Error saving model: ' + error.message, true);
    } finally {
        requestsInProgress--;
        if (saveButton) saveButton.disabled = false;
    }
}

async function loadModelState() {
    if (isTraining || requestsInProgress > 0 || shouldThrottleRequest()) {
        showNotification("Cannot load model right now, please try again in a moment", true);
        return;
    }
    
    lastRequestTime = Date.now();
    if (loadButton) loadButton.disabled = true;
    
    try {
        requestsInProgress++;
        const response = await fetch('/load_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to load model (${response.status}): ${errorText}`);
        }

        await response.json();
        
        setTimeout(() => updateCurrentState(), 1000);
        showNotification('Model loaded successfully');
    } catch (error) {
        console.error("Load model error:", error);
        showNotification('Error loading model: ' + error.message, true);
    } finally {
        requestsInProgress--;
        if (loadButton) loadButton.disabled = false;
    }
}

// Update functions
function updateMetrics(metrics) {
    if (!metrics) return;
    
    if (avgReward) avgReward.textContent = (metrics.average_reward || 0).toFixed(2);
    if (episodeCount) episodeCount.textContent = metrics.episode_count || 0;
    if (successRate) successRate.textContent = `${(metrics.success_rate || 0).toFixed(1)}%`;
    
    // Update progress bar
    if (trainingProgress) {
        const maxEpisodes = metrics.max_episodes || 1000;
        const progress = (metrics.episode_count / maxEpisodes) * 100;
        trainingProgress.style.width = `${Math.min(progress, 100)}%`;
    }
}

function updateVisualization(frameData) {
    if (!frameData || !environmentVisualization) {
        console.warn("No frame data provided or visualization element not found");
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
            
            // Create fallback visualization
            const fallbackDiv = document.createElement('div');
            fallbackDiv.textContent = "Visualization unavailable";
            fallbackDiv.style.width = "100%";
            fallbackDiv.style.height = "100%";
            fallbackDiv.style.display = "flex";
            fallbackDiv.style.alignItems = "center";
            fallbackDiv.style.justifyContent = "center";
            fallbackDiv.style.backgroundColor = "#f0f0f0";
            fallbackDiv.style.color = "#666";
            environmentVisualization.appendChild(fallbackDiv);
        };
        
        // Set source after defining event handlers
        img.src = `data:image/jpeg;base64,${frameData}`;
        img.className = 'w-full h-full object-contain';
        environmentVisualization.appendChild(img);
    } catch (error) {
        console.error("Error in updateVisualization:", error);
    }
}

// Using direct fetch with minimal timeout
async function updateCurrentState() {
    if (requestsInProgress > 0 || shouldThrottleRequest()) {
        console.log("Skipping state update - another request is in progress or throttled");
        return;
    }
    
    lastRequestTime = Date.now();
    try {
        requestsInProgress++;
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // Short 3-second timeout
        
        const response = await fetch('/get_state', {
            signal: controller.signal,
            headers: {
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache'
            }
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to get current state (${response.status}): ${errorText}`);
        }
        
        const data = await response.json();
        updateMetrics(data.metrics);
        if (data.frame) {
            updateVisualization(data.frame);
        }
        
        initialStateLoaded = true;
        
        if (modelStatus && modelStatus.textContent !== 'Training') {
            modelStatus.textContent = 'Model Ready';
            modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
        }
    } catch (error) {
        console.error('Error getting current state:', error);
        
        // Set fallback state if we couldn't get the actual state
        if (!initialStateLoaded) {
            if (modelStatus) {
                if (error.name === 'AbortError') {
                    modelStatus.textContent = 'Connection Timeout';
                } else {
                    modelStatus.textContent = 'Connection Error';
                }
                modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
            }
        }
        
        // Only show notification for critical errors
        if (error.name !== 'AbortError' && !error.message.includes('Failed to fetch')) {
            showNotification('Error getting current state: ' + error.message, true);
        }
    } finally {
        requestsInProgress--;
    }
}

// Notification function with improved positioning and fading
function showNotification(message, isError = false) {
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification-alert');
    existingNotifications.forEach(notification => {
        notification.remove();
    });
    
    const notification = document.createElement('div');
    notification.className = `notification-alert fixed top-4 right-4 px-6 py-3 rounded-lg text-white ${
        isError ? 'bg-red-600' : 'bg-green-600'
    } shadow-lg z-50 transition-opacity duration-300 max-w-md`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Fade out and remove after 5 seconds (longer for errors)
    const displayTime = isError ? 8000 : 3000;
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, displayTime);
}

// Initialize backend health check
async function checkBackendHealth() {
    try {
        // Use a short timeout for the health check
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        
        const response = await fetch('/get_state', { 
            signal: controller.signal,
            headers: { 'Cache-Control': 'no-cache' }
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
            console.log('Backend is healthy');
            return true;
        } else {
            console.warn('Backend health check failed:', response.status);
            return false;
        }
    } catch (error) {
        console.error('Health check error:', error);
        return false;
    }
}

// Event Listeners with guards against null elements
if (startButton) {
    startButton.addEventListener('click', startTraining);
}
if (stopButton) {
    stopButton.addEventListener('click', stopTraining);
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
    mapSizeSelect.addEventListener('change', () => {
        // Don't automatically randomize on change - just update the UI
        console.log(`Map size changed to ${mapSizeSelect.value}x${mapSizeSelect.value}`);
    });
}

// Initialize
if (stopButton) {
    stopButton.disabled = true;
}
console.log('Initializing RL training interface');

// Check backend health before initializing
checkBackendHealth().then(isHealthy => {
    if (isHealthy) {
        // Wait a short time before updating state
        setTimeout(() => {
            updateCurrentState();
        }, 500);
    } else {
        if (modelStatus) {
            modelStatus.textContent = 'Backend Unavailable';
            modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
        }
        showNotification('Backend service is unreachable or slow to respond. Please try again later.', true);
    }
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isTraining) {
        console.log('Page hidden, stopping training');
        stopTraining();
    } else if (!document.hidden && initialStateLoaded) {
        console.log('Page visible, updating state');
        // Add a delay to avoid immediate requests on tab focus
        setTimeout(() => {
            updateCurrentState();
        }, 1000);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    // Stop training if active
    if (isTraining) {
        console.log('Page unloading, stopping training');
        navigator.sendBeacon('/stop_training', JSON.stringify({}));
        isTraining = false;
    }
});