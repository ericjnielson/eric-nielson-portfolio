class TeachingAssistantUI {
    constructor() {
        // Initialize element references
        this.initializeElements();
        
        // Only proceed with setup if required elements exist
        if (this.checkRequiredElements()) {
            this.setupEventListeners();
            this.loadInitialPrompt();
        } else {
            console.error('Required elements not found in the DOM');
        }
    }

    initializeElements() {
        this.weekSelect = document.getElementById('weekSelect');
        this.discussionSelect = document.getElementById('discussionSelect');
        this.postInput = document.getElementById('postInput');
        this.generateButton = document.querySelector('.analyze-button');
        this.feedbackContainer = document.querySelector('.feedback-content');
        this.promptElement = document.getElementById('currentPrompt');
        this.objectivesList = document.getElementById('promptObjectives');
    }
    
    checkRequiredElements() {
        const requiredElements = {
            'Week Select': this.weekSelect,
            'Discussion Select': this.discussionSelect,
            'Post Input': this.postInput,
            'Generate Button': this.generateButton,
            'Feedback Container': this.feedbackContainer,
            'Prompt Element': this.promptElement,
            'Objectives List': this.objectivesList
        };

        let allFound = true;
        for (const [name, element] of Object.entries(requiredElements)) {
            if (!element) {
                console.error(`Required element not found: ${name}`);
                allFound = false;
            }
        }
        
        return allFound;
    }
    
    setupEventListeners() {
        if (this.generateButton) {
            this.generateButton.addEventListener('click', () => this.generateFeedback());
        }
        if (this.weekSelect) {
            this.weekSelect.addEventListener('change', () => this.updatePrompt());
        }
        if (this.discussionSelect) {
            this.discussionSelect.addEventListener('change', () => this.updatePrompt());
        }
    }

    loadInitialPrompt() {
        if (this.weekSelect && this.discussionSelect) {
            this.updatePrompt();
        }
    }
    
    async generateFeedback() {
        if (!this.postInput || !this.postInput.value.trim()) {
            this.showToast('Please enter a student post');
            return;
        }
        
        try {
            this.setLoading(true);
            const feedback = await this.getFeedback();
            if (feedback) {
                this.displayFeedback(feedback);
                this.showToast('Feedback generated successfully!');
            }
        } catch (error) {
            console.error('Error generating feedback:', error);
            const errorMessage = error.message.includes('timed out') ? 
                'Request timed out. Please try a shorter post.' : 
                error.message || 'An unexpected error occurred';
            this.showToast(`Error: ${errorMessage}`);
        } finally {
            this.setLoading(false);
        }
    }
    
    async getFeedback() {
        try {
            if (!this.weekSelect || !this.discussionSelect || !this.postInput) {
                throw new Error('Required form elements not found');
            }
    
            const post = this.postInput.value.trim();
            if (!post) {
                throw new Error('Please enter a student post');
            }
    
            const requestData = {
                week: parseInt(this.weekSelect.value),
                discussion: parseInt(this.discussionSelect.value),
                post: post
            };
    
            console.log('Sending analysis request with data:', requestData);
    
            // Add timeout handling
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 60000); // 60 second timeout
    
            try {
                const response = await fetch('/api/analyze-post', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData),
                    signal: controller.signal
                });
    
                clearTimeout(timeout);
    
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.error || `Server error: ${response.status}`);
                }
    
                const data = await response.json();
                if (!data) {
                    throw new Error('No feedback data received');
                }
    
                return data;
    
            } catch (error) {
                if (error.name === 'AbortError') {
                    throw new Error('Request timed out. Please try again with a shorter post.');
                }
                throw error;
            }
    
        } catch (error) {
            console.error('Feedback request failed:', error);
            throw error;
        }
    }

    async updatePrompt() {
        try {
            if (!this.weekSelect || !this.discussionSelect) {
                throw new Error('Week or discussion select elements not found');
            }

            const week = parseInt(this.weekSelect.value);
            const discussion = parseInt(this.discussionSelect.value);
            
            const response = await fetch(`/api/week-content/${week}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch week content');
            }

            if (!data) {
                throw new Error('No data received from server');
            }

            // Get discussion-specific data
            const discussionData = data[discussion];
            if (!discussionData) {
                throw new Error('Invalid discussion data received');
            }

            // Update title and prompt if they exist
            if (this.promptElement) {
                this.promptElement.textContent = discussionData.prompt || 'No prompt available';
            }

            // Update objectives
            if (this.objectivesList) {
                this.objectivesList.innerHTML = `
                    <h4>Learning Objectives:</h4>
                    ${discussionData.objectives.map(objective => `<li>${objective}</li>`).join('')}
                `;
            }

        } catch (error) {
            console.error('Error fetching prompt:', error);
            this.showToast('Error loading discussion prompt: ' + error.message);
            this.setDefaultContent();
        }
    }

    // Rest of your methods remain the same...
    
    setDefaultContent() {
        if (this.promptElement) {
            this.promptElement.textContent = 'Error loading prompt. Please try again.';
        }
        
        if (this.objectivesList) {
            this.objectivesList.innerHTML = '<li>Error loading objectives</li>';
        }
    }
    
    displayFeedback(feedback) {
        try {
            // Update text feedback sections
            const sections = {
                'positiveFeedback': feedback.positive_feedback || feedback.positive,
                'developmentFeedback': feedback.areas_for_development || feedback.development,
                'connectionsFeedback': feedback.future_connections || feedback.connections
            };

            for (const [id, text] of Object.entries(sections)) {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = text || 'No feedback available';
                }
            }

            // Update metrics
            const metrics = feedback.metrics || {};
            const metricElements = {
                'contentScore': metrics.content_coverage,
                'thinkingScore': metrics.critical_thinking,
                'applicationScore': metrics.practical_application
            };

            for (const [id, value] of Object.entries(metricElements)) {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = value ? `${Math.round(value * 100)}%` : 'N/A';
                }
            }
        } catch (error) {
            console.error('Error displaying feedback:', error);
            this.showToast('Error displaying feedback');
        }
    }

    setLoading(isLoading) {
        if (this.generateButton) {
            this.generateButton.disabled = isLoading;
            this.generateButton.classList.toggle('loading', isLoading);
            this.generateButton.textContent = isLoading ? 'Analyzing...' : 'Generate Feedback';
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.textContent = message;
        
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 10);
        
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// Initialize when document loads
document.addEventListener('DOMContentLoaded', () => {
    window.teachingAssistant = new TeachingAssistantUI();
});