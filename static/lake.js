// Initialize Socket.IO with Cloud Run optimizations
const socket = io({
    path: '/ws/socket.io',
    transports: ['polling', 'websocket'], // Start with polling which is more reliable for initial connection
    reconnection: true,
    reconnectionAttempts: 15,    // Increased for Cloud Run
    reconnectionDelay: 1000,
    reconnectionDelayMax: 10000, // Longer max delay for Cloud Run scaling
    timeout: 30000,              // Longer timeout for cold starts
    forceNew: true,              // Force new connection to avoid stale connections
    query: {                     // Add timestamp to avoid caching issues
        t: Date.now()
    }
});

// Connection state and request management
let socketConnected = false;
let requestInProgress = false;
let lastRequestTime = 0;
const REQUEST_COOLDOWN = 1500;   // Increased cooldown for Cloud Run
const HEALTH_CHECK_INTERVAL = 30000; // Regular health checks every 30s
let healthCheckTimer = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
let initialStateLoaded = false;

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

// Add a health check ping function for Cloud Run
function startHealthCheck() {
    if (healthCheckTimer) clearInterval(healthCheckTimer);
    
    healthCheckTimer = setInterval(() => {
        if (socketConnected) {
            // Send a lightweight ping to keep connection alive
            socket.emit('ping', { timestamp: Date.now() });
        } else if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            reconnectAttempts++;
            console.log(`Attempting to reconnect (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})...`);
            socket.connect();
        }
    }, HEALTH_CHECK_INTERVAL);
}

// Add function to stop health check
function stopHealthCheck() {
    if (healthCheckTimer) {
        clearInterval(healthCheckTimer);
        healthCheckTimer = null;
    }
}

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
    reconnectAttempts = 0;
    
    // Start health check
    startHealthCheck();
    
    // Request initial state update when connected
    if (!initialStateLoaded) {
        // Longer delay for Cloud Run cold start
        setTimeout(() => updateCurrentState(), 2000);
    }
});

socket.on('connect_error', (error) => {
    console.error('Socket connection error:', error);
    if (modelStatus) {
        modelStatus.textContent = 'Connection Error';
        modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800';
    }
    socketConnected = false;
    
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        showNotification('Connection error: Unable to connect to server. The server may be restarting or experiencing high load.', true);
    }
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
        stopTraining(true); // Force stop
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

socket.on('error', (error) => {
    console.error('Socket error:', error);
});

socket.on('pong', () => {
    console.log('Received pong from server - connection active');
});

// Cloud Run optimized fetch with exponential backoff
async function cloudRunFetch(url, options = {}, maxRetries = 3) {
    let retries = 0;
    
    while (retries < maxRetries) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), options.timeout || 10000);
            
            options.signal = controller.signal;
            options.headers = {
                ...options.headers,
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            };
            
            const response = await fetch(url, options);
            clearTimeout(timeoutId);
            
            if (response.status === 429 || response.status >= 500) {
                // Server overloaded or error - retry with backoff
                retries++;
                const backoff = Math.min(1000 * Math.pow(2, retries), 10000); // Exponential backoff
                await new Promise(r => setTimeout(r, backoff));
                continue;
            }
            
            return response;
        } catch (error) {
            if (error.name === 'AbortError' || retries >= maxRetries - 1) {
                throw error;
            }
            
            retries++;
            const backoff = Math.min(1000 * Math.pow(2, retries), 10000);
            await new Promise(r => setTimeout(r, backoff));
        }
    }
    
    throw new Error(`Failed after ${maxRetries} retries`);
}

// Rate limiting helper
function canMakeRequest() {
    const now = Date.now();
    if (requestInProgress || now - lastRequestTime < REQUEST_COOLDOWN) {
        console.log("Request throttled, please wait a moment");
        return false;
    }
    return true;
}

// Training control functions
async function startTraining() {
    if (isTraining) {
        console.log("Training already in progress, ignoring start request");
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
        
        // Prepare request data
        const requestData = {
            learning_rate: parseFloat(learningRateSlider?.value || 0.001),
            discount_factor: parseFloat(discountFactorSlider?.value || 0.99),
            exploration_rate: parseFloat(explorationRateSlider?.value || 0.1)
        };
        
        const response = await cloudRunFetch('/start_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData),
            timeout: 15000  // Longer timeout for training start
        }, 2);  // Only 2 retries for training to avoid duplicated starts
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        await response.json();
        
        // Update UI state
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
        
        // Re-enable the button regardless of the outcome
        if (startButton) startButton.disabled = false;
    } finally {
        requestInProgress = false;
    }
}

async function stopTraining(force = false) {
    if (!isTraining && !force) {
        console.log("No training in progress, ignoring stop request");
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
        
        // Only make the request if not forced or if socketConnected
        if (!force || socketConnected) {
            try {
                const response = await cloudRunFetch('/stop_training', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({}),
                    timeout: 8000  // Shorter timeout for stopping
                }, 2);
                
                if (!response.ok) {
                    console.warn(`Stop training response not OK: ${response.status}`);
                }
            } catch (requestError) {
                console.warn("Error during stop request:", requestError);
                // Continue with UI updates even if request fails
            }
        }
    } catch (error) {
        console.error("Error in stop training:", error);
    } finally {
        // Always update UI state regardless of success/failure
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
        
        // Get selected map size (with validation)
        const mapSize = parseInt(mapSizeSelect?.value || 8);
        
        const response = await cloudRunFetch('/randomize_map', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ size: mapSize }),
            timeout: 10000
        }, 3);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update visualization if we got frame data
        if (data.frame) {
            updateVisualization(data.frame);
        }
        
        // Update metrics if available
        if (data.metrics) {
            updateMetrics(data.metrics);
        }
        
        showNotification("Map randomized successfully");
        
    } catch (error) {
        console.error("Randomize map error:", error);
        showNotification('Error randomizing map: ' + error.message, true);
    } finally {
        requestInProgress = false;
        if (randomizeButton) randomizeButton.disabled = false;
    }
}

async function saveModelState() {
    if (isTraining || !canMakeRequest()) {
        showNotification("Cannot save model right now", true);
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (saveButton) saveButton.disabled = true;
        
        const response = await cloudRunFetch('/save_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
            timeout: 10000
        }, 3);
        
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
    if (isTraining || !canMakeRequest()) {
        showNotification("Cannot load model right now", true);
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        if (loadButton) loadButton.disabled = true;
        
        const response = await cloudRunFetch('/load_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
            timeout: 10000
        }, 3);
        
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
    
    // Update progress bar
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

async function updateCurrentState() {
    if (requestInProgress) {
        console.log("Skipping state update - another request is in progress");
        return;
    }
    
    try {
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        const response = await cloudRunFetch('/get_state', {
            timeout: 5000,  // Short timeout for state updates
        }, 2);  // Fewer retries for state updates
        
        if (!response.ok) {
            throw new Error(`Failed to get state: ${response.status}`);
        }
        
        const data = await response.json();
        
        updateMetrics(data.metrics);
        if (data.frame) {
            updateVisualization(data.frame);
        }
        
        initialStateLoaded = true;
        
    } catch (error) {
        console.error('Error getting current state:', error);
        
        // Only show notification for non-timeout errors
        if (error.name !== 'AbortError') {
            showNotification('Failed to get current state: ' + error.message, true);
        }
    } finally {
        requestInProgress = false;
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
    } shadow-lg z-50 transition-opacity duration-300 opacity-0 max-w-md`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Fade in
    setTimeout(() => {
        notification.style.opacity = '1';
    }, 10);
    
    // Fade out and remove after delay
    const displayTime = isError ? 5000 : 3000;
    setTimeout(() => {
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, displayTime);
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
    // Don't automatically trigger map randomization on change
    mapSizeSelect.addEventListener('change', () => {
        console.log(`Map size selected: ${mapSizeSelect.value}x${mapSizeSelect.value}`);
    });
}

// Initialize
if (stopButton) {
    stopButton.disabled = true;
}
console.log('Initializing RL training interface for Cloud Run');

// On page load, check server health and update initial state
async function checkServerAndInitialize() {
    try {
        const response = await cloudRunFetch('/get_state', {
            timeout: 10000  // Longer timeout for initial cold start
        }, 3);
        
        if (response.ok) {
            console.log('Server is available');
            // Wait longer for Cloud Run initialization
            setTimeout(() => updateCurrentState(), 2000);
        } else {
            console.warn('Server health check failed:', response.status);
            showNotification('Server returned an error. Try reloading the page.', true);
        }
    } catch (error) {
        console.error('Server health check error:', error);
        if (error.name === 'AbortError') {
            showNotification('Server is starting up (cold start). Please wait a moment and try again.', true);
        } else {
            showNotification('Failed to connect to server: ' + error.message, true);
        }
    }
}

// Start the initialization with longer delay for Cloud Run
setTimeout(checkServerAndInitialize, 2000);

// Handle page visibility changes for Cloud Run
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (isTraining) {
            console.log('Page hidden, stopping training');
            stopTraining(true);
        }
        
        // Pause health checks when page is hidden to save resources
        stopHealthCheck();
    } else {
        // Resume health checks when page is visible
        startHealthCheck();
        
        if (initialStateLoaded) {
            console.log('Page visible, updating state');
            // Longer delay for potential Cloud Run cold start
            setTimeout(() => updateCurrentState(), 2000);
        }
    }
});

// Cleanup on page unload for Cloud Run
window.addEventListener('beforeunload', () => {
    // Stop health checks
    stopHealthCheck();
    
    if (isTraining) {
        console.log('Page unloading, stopping training');
        // Use a sync request for unload - more reliable for Cloud Run
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/stop_training', false);  // false makes it synchronous
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(JSON.stringify({}));
    }
});