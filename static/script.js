// Color generation functionality
async function generateColors() {
    const colorInput = document.getElementById('colorInput');
    if (!colorInput) {
        console.error("Color input element not found");
        return;
    }
  
    const colorSuggestions = document.getElementById('colorSuggestions');
    if (!colorSuggestions) {
        console.error("Color suggestions element not found");
        return;
    }
  
    const colorValue = colorInput.value.trim();
  
    if (!colorValue) {
        showError(colorSuggestions, "Please enter a color theme or description");
        return;
    }
    
    // Show loading state
    colorSuggestions.innerHTML = '<div class="loading">Generating colors... This may take a few moments</div>';
  
    try {
        // Longer timeout for AI processing (60 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000);
        
        const response = await fetch('/generate_colors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: colorValue }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
  
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status}. ${errorText}`);
        }
  
        const data = await response.json();
        console.log("Color response:", data);
  
        if (data.colors && data.colors.length > 0) {
            // Create gradient background
            const gradient = `linear-gradient(to right, ${data.colors.join(', ')})`;
            document.body.style.background = gradient;
  
            // Display color swatches
            const colorDivs = data.colors.map(
                (color) => `
                    <div class="color-swatch" style="
                        background-color: ${color}; 
                        padding: 15px; 
                        margin: 5px; 
                        color: ${getContrastColor(color)}; 
                        display: inline-block;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        cursor: pointer;
                        transition: transform 0.2s;
                    "
                    onclick="copyToClipboard('${color}')"
                    title="Click to copy">${color}</div>`
            ).join('');
            
            colorSuggestions.innerHTML = `
                <p class="generated-colors-title">Generated Colors (Click to copy):</p>
                <div class="color-swatches">${colorDivs}</div>`;
        } else {
            showError(colorSuggestions, "No colors were generated. Please try a different description.");
        }
    } catch (error) {
        console.error('Error generating colors:', error);
        
        if (error.name === 'AbortError') {
            showError(colorSuggestions, 
                "The request timed out. The server might be busy processing your request or experiencing issues. " +
                "Please try again with a simpler description or try later.");
        } else {
            showError(colorSuggestions, `Error: ${error.message}`);
        }
    }
  }
  
  // Helper function to determine text color based on background color
  function getContrastColor(hexColor) {
      // Convert hex to RGB
      const r = parseInt(hexColor.slice(1, 3), 16);
      const g = parseInt(hexColor.slice(3, 5), 16);
      const b = parseInt(hexColor.slice(5, 7), 16);
      
      // Calculate luminance - standard formula
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      
      // Return black or white based on background brightness
      return luminance > 0.5 ? '#000000' : '#ffffff';
  }
  
  // Helper function to copy color codes to clipboard
  async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast(`Copied ${text} to clipboard!`);
    } catch (err) {
        console.error('Failed to copy text: ', err);
    }
  }
  
  // Helper function to show toast messages
  function showToast(message) {
      const toastContainer = document.getElementById('toastContainer');
      if (!toastContainer) {
          console.error('Toast container not found');
          return;
      }
      
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.textContent = message;
      
      toastContainer.appendChild(toast);
      
      // Trigger animation
      setTimeout(() => toast.classList.add('show'), 10);
      
      // Remove after 3 seconds
      setTimeout(() => {
          toast.classList.remove('show');
          setTimeout(() => toastContainer.removeChild(toast), 300);
      }, 3000);
  }
  
  // Helper function to show errors
  function showError(element, message) {
    if (!element) {
        console.error("Cannot show error: element is null");
        return;
    }
    element.innerHTML = `<div class="error-message">${message}</div>`;
  }
  
  // Background color change function (if still needed)
  function changeBackgroundColor() {
    const colorInput = document.getElementById('colorInput');
    if (!colorInput) {
        console.error("Color input element not found");
        return;
    }
    
    const colorValue = colorInput.value.trim();
    if (!colorValue) {
        alert('Please enter a valid color!');
        return;
    }
    document.body.style.backgroundColor = colorValue;
  }
  
  // Initialize when DOM is fully loaded
  document.addEventListener('DOMContentLoaded', function() {
      // Check if we're on a page with color input functionality
      const colorGenerateButton = document.querySelector('button[onclick="generateColors()"]');
      if (colorGenerateButton) {
          // Make sure the colorSuggestions div exists
          let colorSuggestions = document.getElementById('colorSuggestions');
          if (!colorSuggestions) {
              console.log("Creating missing colorSuggestions element");
              colorSuggestions = document.createElement('div');
              colorSuggestions.id = 'colorSuggestions';
              colorSuggestions.className = 'color-suggestions';
              // Attempt to insert it after the input group
              const inputGroup = document.querySelector('.input-group');
              if (inputGroup) {
                  inputGroup.parentNode.insertBefore(colorSuggestions, inputGroup.nextSibling);
              } else {
                  // Fallback to appending to the body
                  document.body.appendChild(colorSuggestions);
              }
          }
          
          // Make sure the toast container exists
          let toastContainer = document.getElementById('toastContainer');
          if (!toastContainer) {
              console.log("Creating missing toastContainer element");
              toastContainer = document.createElement('div');
              toastContainer.id = 'toastContainer';
              toastContainer.className = 'toast-container';
              document.body.appendChild(toastContainer);
          }
      }
  });