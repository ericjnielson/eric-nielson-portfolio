// Initialize chart and global variables
let forecastChart = null;
const monthlyLabels = ['0m', '3m', '6m', '9m', '12m'];

document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    setupEventListeners();
});

function initializeChart() {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: monthlyLabels,
            datasets: [
                {
                    label: 'Best Case',
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    data: [0, 0, 0, 0, 0],
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Expected Case',
                    borderColor: '#FFC107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    data: [0, 0, 0, 0, 0],
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Worst Case',
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    data: [0, 0, 0, 0, 0],
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${formatCurrency(context.raw)}`;
                        }
                    },
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    cornerRadius: 6
                },
                legend: {
                    position: 'bottom',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

function setupEventListeners() {
    // Event listeners for model tabs
    document.querySelectorAll('.tab-btn').forEach(button => {
        button.addEventListener('click', function() {
            document.querySelectorAll('.tab-btn').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.model-panel').forEach(panel => panel.classList.remove('active'));
            
            this.classList.add('active');
            const modelPanelId = this.getAttribute('data-model') + '-panel';
            document.getElementById(modelPanelId).classList.add('active');
        });
    });

    // Event listeners for single/range toggle buttons
    document.querySelectorAll('.field-toggle-btn').forEach(button => {
        button.addEventListener('click', function() {
            const fieldName = this.dataset.field;
            const mode = this.dataset.mode;

            // Update toggle buttons
            const toggleGroup = this.closest('.field-toggle');
            toggleGroup.querySelectorAll('.field-toggle-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            // Show/hide appropriate input container
            const singleContainer = document.getElementById(`${fieldName}-single`);
            const rangeContainer = document.getElementById(`${fieldName}-range`);
            
            if (mode === 'single') {
                singleContainer.classList.remove('hidden');
                rangeContainer.classList.add('hidden');
            } else {
                singleContainer.classList.add('hidden');
                rangeContainer.classList.remove('hidden');
            }
        });
    });

    // Calculate button event listener
    const calculateBtn = document.getElementById('calculateBtn');
    if (calculateBtn) {
        calculateBtn.addEventListener('click', calculateForecast);
    }

    // Add input validation listeners
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });

        input.addEventListener('blur', function() {
            validateInput(this, true);
        });
    });

    // Add range validation listeners
    document.querySelectorAll('.range-inputs').forEach(rangeGroup => {
        const minInput = rangeGroup.querySelector('input[id$="Min"]');
        const maxInput = rangeGroup.querySelector('input[id$="Max"]');
        
        if (minInput && maxInput) {
            [minInput, maxInput].forEach(input => {
                input.addEventListener('input', () => validateRange(minInput, maxInput));
                input.addEventListener('blur', () => validateRange(minInput, maxInput, true));
            });
        }
    });
}

function validateInput(input, isBlur = false) {
    try {
        const fieldName = input.id.replace('Min', '').replace('Max', '');
        const value = input.value;
        
        if (!isBlur && value === '') {
            input.classList.remove('input-error');
            return;
        }

        const parsedValue = validateAndParseInput(value, fieldName, getActiveModelType());
        input.classList.remove('input-error');
        input.classList.add('input-success');
    } catch (error) {
        input.classList.add('input-error');
        input.classList.remove('input-success');
        if (isBlur) {
            showError(error.message);
        }
    }
}

function validateRange(minInput, maxInput, isBlur = false) {
    if (!minInput.value || !maxInput.value) return;
    
    const min = parseFloat(minInput.value);
    const max = parseFloat(maxInput.value);
    
    if (!isNaN(min) && !isNaN(max)) {
        if (min > max) {
            minInput.classList.add('input-error');
            maxInput.classList.add('input-error');
            if (isBlur) {
                showError('Minimum value cannot be greater than maximum value');
            }
        } else {
            minInput.classList.remove('input-error');
            maxInput.classList.remove('input-error');
        }
    }
}

function getActiveModelType() {
    const activeTab = document.querySelector('.tab-btn.active');
    return activeTab ? activeTab.dataset.model : null;
}

function validateAndParseInput(value, fieldName, modelType) {
    if (!value && value !== '0') {
        throw new Error('Value is required');
    }

    const numValue = parseFloat(value);
    if (isNaN(numValue)) {
        throw new Error('Must be a valid number');
    }

    // Field-specific validation
    switch(fieldName) {
        case 'confidenceLevel':
            if (numValue < 0 || numValue > 10) {
                throw new Error('Must be between 0 and 10');
            }
            break;
            
        case 'currentNPS':
        case 'targetNPS':
            if (numValue < -100 || numValue > 100) {
                throw new Error('Must be between -100 and 100');
            }
            break;
            
        case 'upgradeRate':
        case 'penetrationRate':
        case 'competitorShare':
        case 'currentChurn':
            if (numValue < 0 || numValue > 100) {
                throw new Error('Must be between 0% and 100%');
            }
            break;
            
        default:
            if (fieldName.includes('Cost') || fieldName.includes('Revenue') || 
                fieldName.includes('Value')) {
                if (numValue < 0) {
                    throw new Error('Cannot be negative');
                }
            }
    }

    return numValue;
}

// Growth pattern generators for different model types
function generateGrowthPattern(modelType, targetValue) {
    switch(modelType) {
        case 'market':
            return generateSCurve(targetValue);
        case 'satisfaction':
            return generateSteppedGrowth(targetValue);
        case 'internal':
        case 'upsell':
        default:
            return generateLinearGrowth(targetValue);
    }
}

function generateSCurve(targetValue) {
    const midpoint = 2;
    const steepness = 1.5;
    
    return monthlyLabels.map((_, index) => {
        const x = (index / (monthlyLabels.length - 1)) * 4 - midpoint;
        const progress = 1 / (1 + Math.exp(-steepness * x));
        return targetValue * progress;
    });
}

function generateSteppedGrowth(targetValue) {
    return monthlyLabels.map((_, index) => {
        const progress = index / (monthlyLabels.length - 1);
        const step = Math.floor(progress * 4) / 4;
        return targetValue * step;
    });
}

function generateLinearGrowth(targetValue) {
    return monthlyLabels.map((_, index) => {
        const progress = index / (monthlyLabels.length - 1);
        return targetValue * progress;
    });
}

function updateChart(result, activeModel) {
    const confidenceLevel = result.confidence_level || 5;
    
    // Generate growth patterns for each case
    const chartData = [
        result.best_case,
        result.expected_case,
        result.worst_case
    ].map(value => generateGrowthPattern(activeModel, value));

    // Update datasets
    forecastChart.data.datasets.forEach((dataset, index) => {
        dataset.data = chartData[index];
        dataset.backgroundColor = dataset.backgroundColor.replace(
            /[\d.]+\)$/,
            `${0.1 + ((10 - confidenceLevel) * 0.02)})`
        );
    });
    
    forecastChart.update();
}

function updateResults(result) {
    document.getElementById('bestCase').textContent = formatCurrency(result.best_case);
    document.getElementById('expectedCase').textContent = formatCurrency(result.expected_case);
    document.getElementById('worstCase').textContent = formatCurrency(result.worst_case);

    const confidenceLevel = parseFloat(result.confidence_level || 5) * 10;
    const errorMargin = result.error_margin ? `±${formatCurrency(result.error_margin)}` : '';
    
    const confidenceText = `${confidenceLevel.toFixed(1)}% Confidence ${errorMargin}`;
    ['best', 'expected', 'worst'].forEach(type => {
        document.getElementById(`${type}Confidence`).textContent = confidenceText;
    });
}

// Helper formatting functions
function formatCurrency(value, options = {}) {
    const {
        minimumFractionDigits = 0,
        maximumFractionDigits = 0,
        currency = 'USD'
    } = options;

    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency,
        minimumFractionDigits,
        maximumFractionDigits
    }).format(value);
}

function formatPercentage(value, decimals = 1) {
    return `${parseFloat(value).toFixed(decimals)}%`;
}

function formatNumber(value, options = {}) {
    const {
        decimals = 0,
        suffix = '',
        prefix = ''
    } = options;

    return `${prefix}${parseFloat(value).toFixed(decimals)}${suffix}`;
}

function showError(message, duration = 5000) {
    const existingAlert = document.querySelector('.alert');
    if (existingAlert) {
        existingAlert.remove();
    }

    const alert = document.createElement('div');
    alert.className = 'alert alert-error';
    
    if (message.includes('\n')) {
        const errorList = document.createElement('ul');
        message.split('\n').forEach(error => {
            const li = document.createElement('li');
            li.textContent = error;
            errorList.appendChild(li);
        });
        alert.appendChild(errorList);
    } else {
        alert.textContent = message;
    }

    const container = document.querySelector('.teaching-container');
    container.insertBefore(alert, container.firstChild);

    setTimeout(() => {
        if (alert && alert.parentNode) {
            alert.remove();
        }
    }, duration);
}

async function calculateForecast() {
    try {
        const activeTab = document.querySelector('.tab-btn.active');
        if (!activeTab) {
            throw new Error('No active model selected');
        }
        
        const activeModel = activeTab.dataset.model;
        showLoading(true);
        
        // Collect and transform data in one step
        const formData = collectFormData();
        
        // Add validation before sending
        const validationErrors = validateFormData(formData, activeModel);
        if (validationErrors.length > 0) {
            throw new Error(validationErrors.join('\n'));
        }

        // Note: We're no longer calling transformDataForModel here
        // since collectFormData already includes the transformation
        
        const response = await fetch(`/api/forecast/${activeModel}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Failed to calculate forecast: ${response.statusText}`);
        }

        const result = await response.json();
        
        updateResults(result);
        updateChart(result, activeModel);
        
        if (result.metrics) {
            updateMetrics(result.metrics, activeModel);
        }
        
        if (formData.scenarios) {
            updateCalculationExplanation(activeModel, formData);
        }
        
        showSuccess('Forecast calculated successfully');
    } catch (error) {
        console.error('Calculation error:', error);
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

function showLoading(isLoading) {
    const calculateBtn = document.getElementById('calculateBtn');
    if (!calculateBtn) return;

    calculateBtn.disabled = isLoading;
    calculateBtn.innerHTML = isLoading ? 
        '<span class="spinner"></span>Calculating...' : 
        'Calculate Forecast';
}

function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success';
    alert.textContent = message;
    
    const container = document.querySelector('.teaching-container');
    container.insertBefore(alert, container.firstChild);

    setTimeout(() => {
        if (alert && alert.parentNode) {
            alert.remove();
        }
    }, 3000);
}

function updateMetrics(metrics, modelType) {
    const metricsContainer = document.getElementById('additionalMetrics');
    if (!metricsContainer) return;

    metricsContainer.innerHTML = '';

    // Format metrics based on type
    const formatMetric = (value, type) => {
        switch(type) {
            case 'currency':
                return formatCurrency(value);
            case 'percentage':
                return formatPercentage(value);
            case 'number':
                return formatNumber(value);
            case 'hours':
                return `${formatNumber(value)} hrs`;
            case 'velocity':
                return `${formatNumber(value)}/qtr`;
            default:
                return value;
        }
    };

    // Define metrics display config based on model type
    const metricConfigs = {
        'internal': [
            { key: 'current_annual_cost', label: 'Current Annual Cost', type: 'currency' },
            { key: 'efficiency_gain_percent', label: 'Efficiency Gain', type: 'percentage' },
            { key: 'annual_savings', label: 'Annual Savings', type: 'currency' },
            { key: 'tool_cost', label: 'Tool Cost', type: 'currency' },
            { key: 'velocity_improvement', label: 'Velocity Improvement', type: 'percentage' },
            { key: 'hours_saved', label: 'Hours Saved Annually', type: 'hours' }
        ],
        'upsell': [
            { key: 'potential_upgrades', label: 'Potential Upgrades', type: 'number' },
            { key: 'gross_revenue', label: 'Gross Revenue', type: 'currency' },
            { key: 'implementation_cost', label: 'Implementation Cost', type: 'currency' },
            { key: 'upgrade_rate', label: 'Effective Upgrade Rate', type: 'percentage' }
        ],
        'market': [
            { key: 'available_market', label: 'Available Market', type: 'number' },
            { key: 'potential_customers', label: 'Potential Customers', type: 'number' },
            { key: 'gross_revenue', label: 'Gross Revenue', type: 'currency' },
            { key: 'effective_penetration', label: 'Effective Penetration', type: 'percentage' },
            { key: 'market_share_remaining', label: 'Available Market Share', type: 'percentage' }
        ],
        'satisfaction': [
            { key: 'nps_improvement', label: 'NPS Improvement', type: 'number' },
            { key: 'direct_revenue_impact', label: 'Direct Revenue Impact', type: 'currency' },
            { key: 'churn_improvement_percentage', label: 'Churn Reduction', type: 'percentage' },
            { key: 'retention_revenue', label: 'Retention Revenue', type: 'currency' },
            { key: 'total_revenue_impact', label: 'Total Revenue Impact', type: 'currency' },
            { key: 'initiative_cost', label: 'Initiative Cost', type: 'currency' }
        ]
    };

    const configs = metricConfigs[modelType] || [];

    // Create metric cards with color coding
    configs.forEach(config => {
        if (metrics[config.key] !== undefined) {
            const metricCard = document.createElement('div');
            metricCard.className = 'metric-card';

            // Add color coding based on metric type
            const colorClass = getMetricColorClass(config.type, metrics[config.key]);
            
            metricCard.innerHTML = `
                <div class="flex flex-col">
                    <h4 class="text-gray-600 text-sm font-medium mb-1">${config.label}</h4>
                    <div class="metric-value ${colorClass} text-lg font-bold">
                        ${formatMetric(metrics[config.key], config.type)}
                    </div>
                </div>
            `;
            metricsContainer.appendChild(metricCard);
        }
    });
}

// Helper function to determine color class based on metric type and value
function getMetricColorClass(type, value) {
    switch(type) {
        case 'percentage':
            if (value > 50) return 'text-green-600';
            if (value > 25) return 'text-yellow-600';
            return 'text-red-600';
        case 'currency':
            if (value > 0) return 'text-green-600';
            if (value === 0) return 'text-gray-600';
            return 'text-red-600';
        case 'hours':
            if (value > 1000) return 'text-green-600';
            if (value > 500) return 'text-yellow-600';
            return 'text-gray-600';
        case 'velocity':
            if (value > 50) return 'text-green-600';
            if (value > 25) return 'text-yellow-600';
            return 'text-gray-600';
        default:
            return 'text-gray-800';
    }
}

function validateFormData(data, modelType) {
    const errors = [];
    
    function validatePositiveNumber(value, fieldName) {
        if (typeof value !== 'number' || isNaN(value) || value < 0) {
            errors.push(`${fieldName} must be a positive number`);
        }
    }

    function validatePercentage(value, fieldName) {
        if (typeof value !== 'number' || isNaN(value) || value < 0 || value > 100) {
            errors.push(`${fieldName} must be between 0 and 100`);
        }
    }

    function validateNPS(value, fieldName) {
        if (typeof value !== 'number' || isNaN(value) || value < -100 || value > 100) {
            errors.push(`${fieldName} must be between -100 and 100`);
        }
    }

    // Common validations
    if (!validateConfidenceScore(data.confidence_level)) {
        errors.push("Confidence level must be between 0 and 10");
    }

    switch(modelType) {
        case 'internal':
            validatePositiveNumber(data.annual_hours, "Annual hours");
            validatePositiveNumber(data.avg_salary, "Hourly rate");
            validatePositiveNumber(data.current_velocity, "Current velocity");
            validatePercentage(data.velocity_increase, "Velocity increase");
            validatePositiveNumber(data.tool_cost, "Tool cost");
            break;

        case 'upsell':
            validatePositiveNumber(data.current_customers, "Current customers");
            validatePercentage(data.upgrade_rate, "Upgrade rate");
            validatePositiveNumber(data.price_increase, "Price increase");
            validatePositiveNumber(data.implementation_cost, "Implementation cost");
            break;

        case 'market':
            validatePositiveNumber(data.total_customers, "Total customers");
            validatePercentage(data.penetration_rate, "Penetration rate");
            validatePositiveNumber(data.contract_value, "Contract value");
            validatePositiveNumber(data.initial_costs, "Initial costs");
            validatePercentage(data.competitor_share, "Competitor share");
            break;

        case 'satisfaction':
            validateNPS(data.current_nps, "Current NPS");
            validateNPS(data.target_nps, "Target NPS");
            validatePercentage(data.current_churn, "Current churn");
            validatePositiveNumber(data.initiative_cost, "Initiative cost");
            validatePositiveNumber(data.revenue_impact, "Revenue impact");

            // Additional validation for NPS improvement
            if (data.target_nps <= data.current_nps) {
                errors.push("Target NPS must be greater than current NPS");
            }
            break;
    }

    return errors;
}

function validateConfidenceScore(score) {
    const numScore = parseFloat(score);
    return !isNaN(numScore) && numScore >= 0 && numScore <= 10;
}

function collectFormData() {
    try {
        const activeTab = document.querySelector('.tab-btn.active');
        if (!activeTab) {
            throw new Error('No active model selected');
        }

        const activeModel = activeTab.dataset.model;
        const modelPanel = document.getElementById(`${activeModel}-panel`);
        let best = {}, worst = {}, expected = {};

        // Collect input values
        modelPanel.querySelectorAll('.input-group').forEach(group => {
            const label = group.querySelector('label').textContent.trim();
            const toggleBtn = group.querySelector('.field-toggle-btn.active');
            const mode = toggleBtn ? toggleBtn.dataset.mode : 'single';
            
            const singleInput = group.querySelector('.single-value-container input');
            if (!singleInput) return;
            
            const fieldName = singleInput.id;

            try {
                if (mode === 'single') {
                    const value = validateAndParseInput(singleInput.value, fieldName, activeModel);
                    best[fieldName] = value;
                    worst[fieldName] = value;
                    expected[fieldName] = value;
                } else {
                    const minInput = group.querySelector('input[id$="Min"]');
                    const maxInput = group.querySelector('input[id$="Max"]');
                    
                    if (!minInput?.value || !maxInput?.value) {
                        throw new Error('Both minimum and maximum values are required');
                    }

                    const min = validateAndParseInput(minInput.value, fieldName, activeModel);
                    const max = validateAndParseInput(maxInput.value, fieldName, activeModel);
                    
                    best[fieldName] = max;
                    worst[fieldName] = min;
                    expected[fieldName] = (min + max) / 2;
                }
            } catch (error) {
                throw new Error(`${label}: ${error.message}`);
            }
        });

        // Add confidence level to all scenarios
        const confidenceLevel = parseFloat(document.querySelector(`#${activeModel}-panel input[id*="confidence"]`)?.value || 5);
        best.confidenceLevel = confidenceLevel;
        worst.confidenceLevel = confidenceLevel;
        expected.confidenceLevel = confidenceLevel;

        // Transform the data based on model type
        let transformedData = {
            confidence_level: confidenceLevel
        };

        switch(activeModel) {
            case 'upsell':
                transformedData = {
                    ...transformedData,
                    current_customers: expected.currentCustomers,
                    upgrade_rate: worst.upgradeRate,         // Use minimum upgrade rate for worst case
                    price_increase: worst.incrementalRevenue,
                    implementation_cost: best.implementationCost, // Use maximum cost for worst case
                    scenarios: {
                        best: {
                            customers: best.currentCustomers,
                            rate: best.upgradeRate,
                            price: best.incrementalRevenue,
                            cost: worst.implementationCost
                        },
                        worst: {
                            customers: worst.currentCustomers,
                            rate: worst.upgradeRate,
                            price: worst.incrementalRevenue,
                            cost: best.implementationCost
                        },
                        expected: {
                            customers: expected.currentCustomers,
                            rate: expected.upgradeRate,
                            price: expected.incrementalRevenue,
                            cost: expected.implementationCost
                        }
                    }
                };
                break;
        case 'upsell':
            transformedData = {
                ...transformedData,
                current_customers: rawData.currentCustomers,
                upgrade_rate: rawData.upgradeRate?.min || rawData.upgradeRate,
                price_increase: rawData.incrementalRevenue?.min || rawData.incrementalRevenue,
                implementation_cost: rawData.implementationCost?.max || rawData.implementationCost,
                
                scenarios: {
                    best: {
                        customers: rawData.currentCustomers,
                        rate: (rawData.upgradeRate?.max || rawData.upgradeRate),
                        price: (rawData.incrementalRevenue?.max || rawData.incrementalRevenue),
                        cost: (rawData.implementationCost?.min || rawData.implementationCost)
                    },
                    worst: {
                        customers: rawData.currentCustomers,
                        rate: (rawData.upgradeRate?.min || rawData.upgradeRate),
                        price: (rawData.incrementalRevenue?.min || rawData.incrementalRevenue),
                        cost: (rawData.implementationCost?.max || rawData.implementationCost)
                    },
                    expected: {
                        customers: rawData.currentCustomers,
                        rate: getExpectedValue(rawData.upgradeRate),
                        price: getExpectedValue(rawData.incrementalRevenue),
                        cost: getExpectedValue(rawData.implementationCost)
                    }
                }
            };
            break;

            case 'internal':
                transformedData = {
                    ...transformedData,
                    annual_hours: rawData.annualHours?.min || rawData.annualHours,
                    avg_salary: rawData.hourlyRate?.min || rawData.hourlyRate,
                    current_velocity: rawData.currentVelocity?.min || rawData.currentVelocity,
                    velocity_increase: rawData.velocityIncrease?.min || rawData.velocityIncrease,
                    tool_cost: rawData.toolCost?.max || rawData.toolCost,
                    
                    scenarios: {
                        best: {
                            hours: (rawData.annualHours?.max || rawData.annualHours),
                            salary: (rawData.hourlyRate?.max || rawData.hourlyRate),
                            velocity: (rawData.currentVelocity?.max || rawData.currentVelocity),
                            improvement: (rawData.velocityIncrease?.max || rawData.velocityIncrease),
                            costs: (rawData.toolCost?.min || rawData.toolCost)
                        },
                        worst: {
                            hours: (rawData.annualHours?.min || rawData.annualHours),
                            salary: (rawData.hourlyRate?.min || rawData.hourlyRate),
                            velocity: (rawData.currentVelocity?.min || rawData.currentVelocity),
                            improvement: (rawData.velocityIncrease?.min || rawData.velocityIncrease),
                            costs: (rawData.toolCost?.max || rawData.toolCost)
                        },
                        expected: {
                            hours: getExpectedValue(rawData.annualHours),
                            salary: getExpectedValue(rawData.hourlyRate),
                            velocity: getExpectedValue(rawData.currentVelocity),
                            improvement: getExpectedValue(rawData.velocityIncrease),
                            costs: getExpectedValue(rawData.toolCost)
                        }
                    }
                };
                break;
    
            case 'satisfaction':
                transformedData = {
                    ...transformedData,
                    current_nps: rawData.currentNPS?.min || rawData.currentNPS,
                    target_nps: rawData.targetNPS?.min || rawData.targetNPS,
                    current_churn: rawData.currentChurn?.max || rawData.currentChurn,
                    initiative_cost: rawData.initiativeCost?.max || rawData.initiativeCost,
                    revenue_impact: rawData.revenueImpact?.min || rawData.revenueImpact,
                    
                    scenarios: {
                        best: {
                            current_nps: (rawData.currentNPS?.max || rawData.currentNPS),
                            target_nps: (rawData.targetNPS?.max || rawData.targetNPS),
                            current_churn: (rawData.currentChurn?.min || rawData.currentChurn),
                            cost: (rawData.initiativeCost?.min || rawData.initiativeCost),
                            revenue_impact: (rawData.revenueImpact?.max || rawData.revenueImpact)
                        },
                        worst: {
                            current_nps: (rawData.currentNPS?.min || rawData.currentNPS),
                            target_nps: (rawData.targetNPS?.min || rawData.targetNPS),
                            current_churn: (rawData.currentChurn?.max || rawData.currentChurn),
                            cost: (rawData.initiativeCost?.max || rawData.initiativeCost),
                            revenue_impact: (rawData.revenueImpact?.min || rawData.revenueImpact)
                        },
                        expected: {
                            current_nps: getExpectedValue(rawData.currentNPS),
                            target_nps: getExpectedValue(rawData.targetNPS),
                            current_churn: getExpectedValue(rawData.currentChurn),
                            cost: getExpectedValue(rawData.initiativeCost),
                            revenue_impact: getExpectedValue(rawData.revenueImpact)
                        }
                    }
                };
                break;

            case 'market':
                transformedData = {
                    ...transformedData,
                    total_customers: worst.totalCustomers,
                    penetration_rate: worst.penetrationRate,
                    contract_value: worst.contractValue,
                    initial_costs: best.initialCosts,
                    competitor_share: best.competitorShare,
                    market_readiness: worst.marketReadiness,
                    scenarios: {
                        best: {
                            customers: best.totalCustomers,
                            rate: best.penetrationRate,
                            value: best.contractValue,
                            costs: worst.initialCosts,
                            competition: worst.competitorShare
                        },
                        worst: {
                            customers: worst.totalCustomers,
                            rate: worst.penetrationRate,
                            value: worst.contractValue,
                            costs: best.initialCosts,
                            competition: best.competitorShare
                        },
                        expected: {
                            customers: expected.totalCustomers,
                            rate: expected.penetrationRate,
                            value: expected.contractValue,
                            costs: expected.initialCosts,
                            competition: expected.competitorShare
                        }
                    }
                };
                break;
        }

        console.log('Raw scenarios:', { best, worst, expected });
        console.log('Transformed data:', transformedData);
        return transformedData;

    } catch (error) {
        showError(error.message);
        throw error;
    }
}

// Helper function to get appropriate value from range
function getRangeValue(value, type = 'expected') {
    if (!value) return 0;
    
    // If it's a single value
    if (typeof value === 'number') return value;
    
    // If it's a range
    if (value.min !== undefined && value.max !== undefined) {
        switch(type) {
            case 'best':
                return value.max;
            case 'worst':
                return value.min;
            case 'expected':
            default:
                return (value.min + value.max) / 2;
        }
    }
    
    return 0;
}

function getExpectedValue(value) {
    if (!value) return 0;
    if (typeof value === 'number') return value;
    if (value.min !== undefined && value.max !== undefined) {
        return (value.min + value.max) / 2;
    }
    return value;
}

function updateCalculationExplanation(modelType, data) {
    const explanationContainer = document.getElementById('calculationExplanation');
    if (!explanationContainer) return;

    const getExplanationContent = (type, data) => {
        switch(type) {
            case 'upsell':
                return `
                    <div class="calculation-explanation">
                        <h2>How These Numbers Are Calculated</h2>
                        
                        <div class="scenario-card best-case">
                            <h3>Best Case Scenario</h3>
                            <p>Formula: (Current Customers × Highest Upgrade Rate × Increased Revenue) - Lowest Implementation Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.best.customers)} × ${data.scenarios.best.rate}% × ${formatCurrency(data.scenarios.best.price)}) - ${formatCurrency(data.scenarios.best.cost)}</code>
                            </div>
                        </div>

                        <div class="scenario-card expected-case">
                            <h3>Expected Case Scenario</h3>
                            <p>Formula: (Current Customers × Average Upgrade Rate × Increased Revenue) - Average Implementation Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.expected.customers)} × ${data.scenarios.expected.rate}% × ${formatCurrency(data.scenarios.expected.price)}) - ${formatCurrency(data.scenarios.expected.cost)}</code>
                            </div>
                        </div>

                        <div class="scenario-card worst-case">
                            <h3>Worst Case Scenario</h3>
                            <p>Formula: (Current Customers × Lowest Upgrade Rate × Increased Revenue) - Highest Implementation Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.worst.customers)} × ${data.scenarios.worst.rate}% × ${formatCurrency(data.scenarios.worst.price)}) - ${formatCurrency(data.scenarios.worst.cost)}</code>
                            </div>
                        </div>

                        <div class="insights-grid">
                            <div class="insight-card">
                                <h4>Customer Impact</h4>
                                <ul>
                                    <li>Customer base affects total potential</li>
                                    <li>Upgrade rate drives adoption</li>
                                    <li>Revenue increase per customer</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Revenue Factors</h4>
                                <ul>
                                    <li>Implementation costs affect margin</li>
                                    <li>Upgrade timing impacts cash flow</li>
                                    <li>Customer segmentation benefits</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Risk Considerations</h4>
                                <ul>
                                    <li>Adoption rate uncertainty</li>
                                    <li>Implementation complexity</li>
                                    <li>Customer satisfaction impact</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                `;

            case 'market':
                return `
                    <div class="calculation-explanation">
                        <h2>How These Numbers Are Calculated</h2>
                        
                        <div class="scenario-card best-case">
                            <h3>Best Case Scenario</h3>
                            <p>Formula: (Total Market × (1 - Competition Rate) × Market Penetration × Contract Value) - Initial Costs</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.best.customers)} × (1 - ${data.scenarios.best.competition}%) × ${data.scenarios.best.rate}% × ${formatCurrency(data.scenarios.best.value)}) - ${formatCurrency(data.scenarios.best.costs)}</code>
                            </div>
                        </div>

                        <div class="scenario-card expected-case">
                            <h3>Expected Case Scenario</h3>
                            <p>Formula: (Total Market × (1 - Competition Rate) × Market Penetration × Contract Value) - Initial Costs</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.expected.customers)} × (1 - ${data.scenarios.expected.competition}%) × ${data.scenarios.expected.rate}% × ${formatCurrency(data.scenarios.expected.value)}) - ${formatCurrency(data.scenarios.expected.costs)}</code>
                            </div>
                        </div>

                        <div class="scenario-card worst-case">
                            <h3>Worst Case Scenario</h3>
                            <p>Formula: (Total Market × (1 - Competition Rate) × Market Penetration × Contract Value) - Initial Costs</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>(${formatNumber(data.scenarios.worst.customers)} × (1 - ${data.scenarios.worst.competition}%) × ${data.scenarios.worst.rate}% × ${formatCurrency(data.scenarios.worst.value)}) - ${formatCurrency(data.scenarios.worst.costs)}</code>
                            </div>
                        </div>

                        <div class="insights-grid">
                            <div class="insight-card">
                                <h4>Market Factors</h4>
                                <ul>
                                    <li>Market size potential</li>
                                    <li>Competitive landscape</li>
                                    <li>Penetration opportunities</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Revenue Impact</h4>
                                <ul>
                                    <li>Contract value per customer</li>
                                    <li>Initial investment required</li>
                                    <li>Market timing considerations</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Risk Factors</h4>
                                <ul>
                                    <li>Competition response</li>
                                    <li>Market entry barriers</li>
                                    <li>Adoption timeline</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                `;

            case 'internal':
                return `
                    <div class="calculation-explanation">
                        <h2>How These Numbers Are Calculated</h2>
                        
                        <div class="scenario-card best-case">
                            <h3>Best Case Scenario</h3>
                            <p>Formula: ((Annual Hours × Hourly Rate) × Velocity Improvement) - Tool Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${formatNumber(data.scenarios.best.hours)} × ${formatCurrency(data.scenarios.best.salary)}) × ${data.scenarios.best.improvement}%) - ${formatCurrency(data.scenarios.best.costs)}</code>
                            </div>
                        </div>

                        <div class="scenario-card expected-case">
                            <h3>Expected Case Scenario</h3>
                            <p>Formula: ((Annual Hours × Hourly Rate) × Velocity Improvement) - Tool Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${formatNumber(data.scenarios.expected.hours)} × ${formatCurrency(data.scenarios.expected.salary)}) × ${data.scenarios.expected.improvement}%) - ${formatCurrency(data.scenarios.expected.costs)}</code>
                            </div>
                        </div>

                        <div class="scenario-card worst-case">
                            <h3>Worst Case Scenario</h3>
                            <p>Formula: ((Annual Hours × Hourly Rate) × Velocity Improvement) - Tool Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${formatNumber(data.scenarios.worst.hours)} × ${formatCurrency(data.scenarios.worst.salary)}) × ${data.scenarios.worst.improvement}%) - ${formatCurrency(data.scenarios.worst.costs)}</code>
                            </div>
                        </div>

                        <div class="insights-grid">
                            <div class="insight-card">
                                <h4>Efficiency Factors</h4>
                                <ul>
                                    <li>Resource utilization</li>
                                    <li>Process improvements</li>
                                    <li>Time savings</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Cost Impact</h4>
                                <ul>
                                    <li>Tool implementation</li>
                                    <li>Training requirements</li>
                                    <li>Maintenance costs</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Risk Factors</h4>
                                <ul>
                                    <li>Adoption challenges</li>
                                    <li>Integration complexity</li>
                                    <li>Process changes</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                `;

            case 'satisfaction':
                return `
                    <div class="calculation-explanation">
                        <h2>How These Numbers Are Calculated</h2>
                        
                        <div class="scenario-card best-case">
                            <h3>Best Case Scenario</h3>
                            <p>Formula: ((Target NPS - Current NPS) × Revenue Impact × (1 - Churn Rate)) - Initiative Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${data.scenarios.best.target_nps} - ${data.scenarios.best.current_nps}) × ${formatCurrency(data.scenarios.best.revenue_impact)} × (1 - ${data.scenarios.best.current_churn}%)) - ${formatCurrency(data.scenarios.best.cost)}</code>
                            </div>
                        </div>

                        <div class="scenario-card expected-case">
                            <h3>Expected Case Scenario</h3>
                            <p>Formula: ((Target NPS - Current NPS) × Revenue Impact × (1 - Churn Rate)) - Initiative Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${data.scenarios.expected.target_nps} - ${data.scenarios.expected.current_nps}) × ${formatCurrency(data.scenarios.expected.revenue_impact)} × (1 - ${data.scenarios.expected.current_churn}%)) - ${formatCurrency(data.scenarios.expected.cost)}</code>
                            </div>
                        </div>

                        <div class="scenario-card worst-case">
                            <h3>Worst Case Scenario</h3>
                            <p>Formula: ((Target NPS - Current NPS) × Revenue Impact × (1 - Churn Rate)) - Initiative Cost</p>
                            <div class="formula-block">
                                <p>Using your numbers:</p>
                                <code>((${data.scenarios.worst.target_nps} - ${data.scenarios.worst.current_nps}) × ${formatCurrency(data.scenarios.worst.revenue_impact)} × (1 - ${data.scenarios.worst.current_churn}%)) - ${formatCurrency(data.scenarios.worst.cost)}</code>
                            </div>
                        </div>

                        <div class="insights-grid">
                            <div class="insight-card">
                                <h4>Satisfaction Impact</h4>
                                <ul>
                                    <li>NPS improvement potential</li>
                                    <li>Revenue correlation</li>
                                    <li>Churn reduction</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Financial Impact</h4>
                                <ul>
                                    <li>Initiative investment</li>
                                    <li>Revenue preservation</li>
                                    <li>Customer lifetime value</li>
                                </ul>
                            </div>
                            
                            <div class="insight-card">
                                <h4>Risk Factors</h4>
                                <ul>
                                    <li>Implementation effectiveness</li>
                                    <li>Customer response</li>
                                    <li>Competitive factors</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                `;

            default:
                return '';
        }
    };

    explanationContainer.innerHTML = getExplanationContent(modelType, data);
}