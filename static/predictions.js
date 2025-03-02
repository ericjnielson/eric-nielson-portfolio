// predictions.js
async function loadTeams() {
    try {
        const response = await fetch('/api/teams');
        if (!response.ok) throw new Error('Failed to fetch teams');
        const teams = await response.json();
        
        // Define Power Five conferences and their teams
        const powerFiveTeams = {
            'SEC': [
                'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 
                'Kentucky', 'LSU', 'Mississippi State', 'Missouri', 
                'Ole Miss', 'South Carolina', 'Tennessee', 'Texas A&M', 'Vanderbilt', 
                'Oklahoma', 'Texas' 
            ],
            'Big Ten': [
                'Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan', 
                'Michigan State', 'Minnesota', 'Nebraska', 'Northwestern', 
                'Ohio State', 'Penn State', 'Purdue', 'Rutgers', 'Wisconsin', 
                'UCLA', 'USC', 'Oregon', 'Washington' 
            ],
            'ACC': [
                'Boston College', 'Clemson', 'Duke', 'Florida State', 
                'Georgia Tech', 'Louisville', 'Miami', 'NC State', 
                'North Carolina', 'Pittsburgh', 'Syracuse', 'Virginia', 
                'Virginia Tech', 'Wake Forest', 
                'California', 'Stanford'  
            ],
            'Big 12': [
                'Baylor', 'BYU', 'Cincinnati', 'Colorado', 'Houston', 'Iowa State', 
                'Kansas', 'Kansas State', 'Oklahoma State', 'TCU', 
                'Texas Tech', 'UCF', 'West Virginia' 
            ],
            'Pac-12': [
                'Arizona State', 'Colorado State', 'Oregon State',  
                'San Diego State', 'SMU', 'Utah', 'Utah State', 'Washington State' 
            ],
            'Mountain West': [
                'Air Force', 'Boise State', 'Fresno State', 'Hawaii', 'Nevada', 
                'New Mexico', 'San Jose State', 'UNLV', 'Wyoming' 
            ],
            'Independents': [
                'Army', 'BYU', 'Connecticut', 'Massachusetts', 'New Mexico State', 
                'Notre Dame'
            ]
        };

        // Populate conference dropdowns
        const homeConference = document.getElementById('homeConference');
        const awayConference = document.getElementById('awayConference');
        
        if (!homeConference || !awayConference) {
            console.error('Conference select elements not found');
            return;
        }
        
        // Reset dropdowns
        homeConference.innerHTML = '<option value="">Select Conference</option>';
        awayConference.innerHTML = '<option value="">Select Conference</option>';
        
        // Add conferences in order
        Object.keys(powerFiveTeams).forEach(conference => {
            homeConference.add(new Option(conference, conference));
            awayConference.add(new Option(conference, conference));
        });

        // Event listeners for conference selection
        homeConference.addEventListener('change', () => {
            const homeTeam = document.getElementById('homeTeam');
            if (!homeTeam) return;
            
            homeTeam.innerHTML = '<option value="">Select Team</option>';
            homeTeam.disabled = !homeConference.value;
            
            if (homeConference.value) {
                powerFiveTeams[homeConference.value].sort().forEach(team => {
                    homeTeam.add(new Option(team, team));
                });
            }
        });

        awayConference.addEventListener('change', () => {
            const awayTeam = document.getElementById('awayTeam');
            if (!awayTeam) return;
            
            awayTeam.innerHTML = '<option value="">Select Team</option>';
            awayTeam.disabled = !awayConference.value;
            
            if (awayConference.value) {
                powerFiveTeams[awayConference.value].sort().forEach(team => {
                    awayTeam.add(new Option(team, team));
                });
            }
        });

    } catch (error) {
        console.error('Error loading teams:', error);
        const predictionForm = document.getElementById('predictionForm');
        if (predictionForm) {
            showError(predictionForm, 'Error loading teams. Please refresh the page.');
        }
    }
}

function createErrorContainer() {
    const container = document.createElement('div');
    container.id = 'errorContainer';
    container.className = 'error-message';
    container.style.display = 'none';
    document.getElementById('predictionForm').appendChild(container);
    return container;
}

// Helper function to show errors
function showError(message) {
    const errorContainer = document.getElementById('errorContainer') || 
                          createErrorContainer();
    errorContainer.textContent = message;
    errorContainer.style.display = 'block';
}

async function getPrediction(homeTeam, awayTeam) {
    try {
        console.log('Sending prediction request for:', { homeTeam, awayTeam });
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ homeTeam, awayTeam })
        });

        // Check if response is JSON
        const contentType = response.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            // Get the text response for debugging
            const text = await response.text();
            console.error('Received non-JSON response:', text);
            throw new Error('Server returned non-JSON response');
        }

        const responseData = await response.json();
        console.log('Server response:', responseData);

        if (!response.ok) {
            throw new Error(responseData.error || 'Failed to get prediction');
        }

        return responseData;
    } catch (error) {
        console.error('Prediction error details:', {
            message: error.message,
            stack: error.stack
        });
        throw new Error(`Prediction failed: ${error.message}`);
    }
}

async function updatePrediction() {
    const homeTeam = document.getElementById('homeTeam')?.value;
    const awayTeam = document.getElementById('awayTeam')?.value;
    const resultsContainer = document.getElementById('resultsContainer');
    const errorContainer = document.getElementById('errorContainer') || 
                          createErrorContainer();

    if (!homeTeam || !awayTeam || !resultsContainer) {
        showError('Please select both teams');
        return;
    }

    try {
        resultsContainer.classList.add('loading');
        errorContainer.style.display = 'none';

        const prediction = await getPrediction(homeTeam, awayTeam);
        
        // Update scores and team names
        updateElement('homeTeamName', prediction.homeTeam.name);
        updateElement('awayTeamName', prediction.awayTeam.name);
        updateElement('homeScore', Math.round(prediction.homeTeam.predictedScore));
        updateElement('awayScore', Math.round(prediction.awayTeam.predictedScore));

        // Update prediction metrics
        if (prediction.prediction) {
            updateElement('predictedFavorite', prediction.prediction.favorite);
            const spreadValue = Math.round(prediction.prediction.spread);
            const formattedSpread = prediction.prediction.favorite === prediction.homeTeam.name ? 
                formatValue(spreadValue) : formatValue(spreadValue);
            updateElement('predictedSpread', formattedSpread);
            updateElement('predictedTotal', Math.round(prediction.prediction.total));

            // Update model weights
            updateWeights(prediction.prediction.weights);
        }

        resultsContainer.style.display = 'block';
    } catch (error) {
        console.error('Error in updatePrediction:', error);
        showError(error.message);
    } finally {
        resultsContainer.classList.remove('loading');
    }
}

// Update formatValue function to not use decimals
function formatValue(value) {
    if (value === null || value === undefined) return '--';
    if (value === 0) return "0";
    return value > 0 ? `+${value}` : value.toString();
}

// Helper function to safely update elements and handle undefined values
function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value ?? '--';
    }
}

// Helper function to format values with sign and handle undefined
function formatValue(value) {
    if (value === null || value === undefined) return '--';
    if (value === 0) return "0";
    return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
}

function updateWeights(weights) {
    const weightsContainer = document.getElementById('weightsContainer');
    if (weightsContainer && weights) {
        weightsContainer.innerHTML = Object.entries(weights)
            .map(([factor, weight]) => `
                <div class="weight-item">
                    <span class="weight-label">${formatFactorName(factor)}:</span>
                    <span class="weight-value">${formatWeight(weight)}</span>
                </div>
            `).join('');
    }
}

// Helper function to format factor names
function formatFactorName(factor) {
    return factor
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Helper function to format weight values
function formatWeight(weight) {
    return typeof weight === 'number' ? 
        `${(weight * 100).toFixed(1)}%` : 
        weight.toString();
}

// Initialize only if we're on the predictions page
if (document.getElementById('predictionForm')) {
    document.addEventListener('DOMContentLoaded', function() {
        loadTeams();

        const homeTeam = document.getElementById('homeTeam');
        const awayTeam = document.getElementById('awayTeam');
        const predictionForm = document.getElementById('predictionForm');

        if (homeTeam) homeTeam.addEventListener('change', updatePrediction);
        if (awayTeam) awayTeam.addEventListener('change', updatePrediction);
        if (predictionForm) {
            predictionForm.onsubmit = function(e) {
                e.preventDefault();
                updatePrediction();
            };
        }
    });
}