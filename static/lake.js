// Initialize state tracking variables
let isTraining = false;
let requestInProgress = false;
let lastRequestTime = 0;
let pollingInterval = null;
const POLLING_INTERVAL_MS = 1000; // Poll every second
const MIN_REQUEST_INTERVAL_MS = 500; // Minimum time between requests

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

// Rate limiting for API requests
function canMakeRequest() {
    const now = Date.now();
    return !requestInProgress && (now - lastRequestTime) > MIN_REQUEST_INTERVAL_MS;
}

// Function to start polling for updates
function startPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(async () => {
        if (isTraining && canMakeRequest()) {
            try {
                requestInProgress = true;
                lastRequestTime = Date.now();
                
                const response = await fetch('/get_state', {
                    headers: {
                        'Cache-Control': 'no-cache',
                        'Pragma': 'no-cache'
                    }
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Handle training data update
                    if (data.metrics) {
                        updateMetrics(data.metrics);
                        
                        // Check if training is complete
                        if (data.metrics.training_complete) {
                            console.log('Training complete (from polling)');
                            stopPolling();
                            stopTraining();
                            showNotification('Training complete! ' + 
                                `Success rate: ${data.metrics.success_rate.toFixed(1)}%, ` +
                                `Episodes: ${data.metrics.episode_count}`);
                        }
                    }
                    
                    if (data.frame) {
                        updateVisualization(data.frame);
                    }
                }
                
            } catch (error) {
                console.error('Error during polling:', error);
            } finally {
                requestInProgress = false;
            }
        }
    }, POLLING_INTERVAL_MS);
    
    console.log('Started polling for training updates');
}

// Function to stop polling
function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log('Stopped polling for training updates');
    }
}

// Training control functions
async function startTraining() {
    try {
        console.log("Starting training...");
        
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        const response = await fetch('/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                learning_rate: parseFloat(learningRateSlider.value),
                discount_factor: parseFloat(discountFactorSlider.value),
                exploration_rate: parseFloat(explorationRateSlider.value)
            })
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
        
        // Start polling for updates
        startPolling();

    } catch (error) {
        console.error("Start training error:", error);
        showNotification('Error starting training: ' + error.message, true);
    } finally {
        requestInProgress = false;
    }
}

async function stopTraining() {
    try {
        console.log("Stopping training...");
        
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
        
        const response = await fetch('/stop_training', {
            method: 'POST'
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
        
        // Stop polling
        stopPolling();

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
        
        // Stop polling on error
        stopPolling();
    } finally {
        requestInProgress = false;
    }
}

async function randomizeMap() {
    try {
        console.log("Randomizing map...");
        
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
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
        requestInProgress = false;
    }
}

async function saveModelState() {
    try {
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
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
        requestInProgress = false;
    }
}

async function loadModelState() {
    try {
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
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
        requestInProgress = false;
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
        if (requestInProgress) return;
        requestInProgress = true;
        lastRequestTime = Date.now();
        
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
    } finally {
        requestInProgress = false;
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

// Initialize the page state
function initializePage() {
    console.log('Initializing RL training interface');
    modelStatus.textContent = 'Model Ready';
    modelStatus.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800';
    stopButton.disabled = true;
    updateCurrentState();
}

// Event Listeners
startButton.addEventListener('click', startTraining);
stopButton.addEventListener('click', stopTraining);
saveButton.addEventListener('click', saveModelState);
loadButton.addEventListener('click', loadModelState);
randomizeButton.addEventListener('click', randomizeMap);
mapSizeSelect.addEventListener('change', randomizeMap);

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && isTraining) {
        console.log('Page hidden, stopping polling but keeping training active');
        stopPolling();
    } else if (!document.hidden) {
        console.log('Page visible, updating state');
        if (isTraining) {
            console.log('Resuming polling');
            startPolling();
        }
        updateCurrentState();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (isTraining) {
        console.log('Page unloading, stopping training');
        navigator.sendBeacon('/stop_training', '{}');
        isTraining = false;
        stopPolling();
    }
});

// Initialize the page
initializePage();