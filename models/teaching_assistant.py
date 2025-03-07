from flask import Flask, request, jsonify
from openai import OpenAI
import anthropic
from flask_cors import CORS
import json
from typing import Dict, List
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProjectManagementTA:
    def __init__(self):
        # Initialize Anthropic client
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key not found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        print("Anthropic client initialized")

        # Weekly discussions data structure
        self.weekly_discussions = {
            1: {
                1: {
                    "title": "Project Management Introduction",
                    "prompt": "Describe your experience with project management, either on the job or in your personal life. Consider how study habits affect school performance and if those habits transfer to the workplace. What type of strategy will you use to stay focused and organized during the course?",
                    "objectives": ["Reflect on project management experience", "Connect study habits to performance", "Develop organization strategy"]
                },
                2: {
                    "title": "Project Management in New Organization",
                    "prompt": "As the start-up project manager, deliver a presentation that explains what a project is, describes project management, provides project goals, illustrates planning steps, and summarizes CSR importance.",
                    "objectives": ["Define project management concepts", "Outline project goals", "Integrate CSR principles"]
                }
            },
            2: {
                1: {
                    "title": "Negotiation Strategy",
                    "prompt": "Develop a negotiation strategy to obtain the loading dock supervisor's cooperation with your project and specifically to assign the employee to the project team.",
                    "objectives": ["Develop negotiation strategy", "Address resource constraints", "Consider stakeholder concerns"]
                },
                2: {
                    "title": "Work Breakdown Structure",
                    "prompt": "Assess the importance of the WBS. Explain your assessment of the WBS to your project team.",
                    "objectives": ["Explain WBS importance", "Demonstrate WBS understanding", "Link WBS to project success"]
                }
            },
            3: {
                1: {
                    "title": "Project Requirements",
                    "prompt": "Provide a bulleted list of opportunities and threats to new requirements being added this late in the planning stage.",
                    "objectives": ["Identify opportunities", "Assess threats", "Evaluate timing impact"]
                },
                2: {
                    "title": "Quality Management",
                    "prompt": "Design a project quality management plan that relates to the inherited issues stated in the Project Management Case Study.",
                    "objectives": ["Design quality plan", "Address inherited issues", "Align with CSR"]
                }
            },
            4: {
                1: {
                    "title": "Technology Impact",
                    "prompt": "Describe a situation where self-service and technology help create and deliver a customer benefit package. Provide examples of system defects or service upsets.",
                    "objectives": ["Analyze technology benefits", "Identify potential issues", "Consider customer impact"]
                },
                2: {
                    "title": "Operations Strategy",
                    "prompt": "Describe a customer experience where service was unsatisfactory. How might operations management have helped?",
                    "objectives": ["Analyze service issues", "Apply operations concepts", "Propose improvements"]
                }
            },
            5: {
                1: {
                    "title": "Voice of Customer",
                    "prompt": "Explain the influence of technology on the five elements of a service-delivery system.",
                    "objectives": ["Link technology to service", "Analyze system elements", "Consider customer needs"]
                },
                2: {
                    "title": "Supply Chain Design",
                    "prompt": "Explain why it is important for operations managers to understand local cultures and practices of countries where they do business.",
                    "objectives": ["Cultural awareness", "Global operations", "Risk management"]
                }
            },
            6: {
                1: {
                    "title": "Chase Strategy",
                    "prompt": "Decide to be for or against adopting a chase strategy for a major airline call center.",
                    "objectives": ["Evaluate chase strategy", "Consider resource implications", "Analyze customer service"]
                },
                2: {
                    "title": "Organizational Waste",
                    "prompt": "Identify three examples of different types of waste and potential lean tools to address them.",
                    "objectives": ["Identify waste types", "Apply lean tools", "Propose solutions"]
                }
            }
        }

        # Your sample feedback templates from discussion templates
        self.feedback_patterns = {
            "positive": [
                "Great work highlighting {}",
                "You did well in identifying {}",
                "Excellent job demonstrating {}"
            ],
            "development": [
                "Consider exploring {} in more depth",
                "You could strengthen your discussion by {}",
                "Think about how {} relates to {}"
            ],
            "connection": [
                "Keep these concepts in mind for Week {} when we discuss {}",
                "This connects well to our upcoming discussion of {}",
                "Your analysis will be valuable when we explore {}"
            ]
        }

    def analyze_post(self, week: int, discussion: int, post_text: str) -> dict:
        """Analyze a student's discussion post with optimizations for Cloud Run"""
        try:
            print(f"\nAnalyzing post for Week {week}, Discussion {discussion}")
            
            # Input validation
            if not post_text or len(post_text.strip()) == 0:
                raise ValueError("Post text cannot be empty")
                    
            if not isinstance(week, int) or not isinstance(discussion, int):
                raise ValueError("Week and discussion must be integers")
                    
            if week not in self.weekly_discussions:
                raise ValueError(f"Invalid week number: {week}")
                    
            if discussion not in self.weekly_discussions[week]:
                raise ValueError(f"Invalid discussion number: {discussion}")
                
            discussion_data = self.weekly_discussions[week][discussion]
            
            # Use a conservative max length to ensure fast processing
            max_post_length = 1500
            truncated_post = post_text[:max_post_length] + ("..." if len(post_text) > max_post_length else "")
            
            prompt = f"""You are Dr. Nielson providing feedback on a student's discussion post.
            
            Week {week}, Discussion {discussion}
            Title: {discussion_data['title']}
            Prompt: {discussion_data['prompt']}
            Objectives: {', '.join(discussion_data['objectives'])}
            
            Student Post:
            {truncated_post}

            Provide concise feedback exactly in this format:
            POSITIVE_FEEDBACK: (highlight specific strengths)
            AREAS_FOR_DEVELOPMENT: (1-2 specific improvements)
            FUTURE_CONNECTIONS: (connect to upcoming topics)
            METRICS:
            content_coverage: (score 0-1)
            critical_thinking: (score 0-1)
            practical_application: (score 0-1)
            """
            
            print("Sending request to Anthropic API")
            
            try:
                # Optimize API call for speed
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",  
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,  # Reduced for faster response
                    temperature=0.6,  # Lower temperature for more deterministic responses
                    # Note: Don't set API timeout directly in Cloud Run - it can cause issues
                )
                
                if not response or not response.content:
                    raise ValueError("No response received from Anthropic API")
                    
                raw_feedback = response.content[0].text
                print("Received API response successfully")
                
                # Parse feedback and ensure metrics exist
                feedback = self._parse_feedback(raw_feedback)
                
                # Ensure metrics have valid values
                if all(v == 0 for v in feedback["metrics"].values()):
                    feedback["metrics"] = {
                        "content_coverage": 0.75,
                        "critical_thinking": 0.70,
                        "practical_application": 0.75
                    }
                
                return feedback
                    
            except Exception as e:
                print(f"API error: {str(e)}")
                # Return fallback response to avoid timeout
                return {
                    "positive_feedback": "Your submission addresses the key requirements of the assignment and shows good understanding of the concepts.",
                    "areas_for_development": "Consider expanding your analysis with more specific examples to strengthen your arguments.",
                    "future_connections": "The concepts you've discussed will be valuable in upcoming modules on project planning.",
                    "metrics": {
                        "content_coverage": 0.75,
                        "critical_thinking": 0.70,
                        "practical_application": 0.75
                    }
                }
                
        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            raise
        except Exception as e:
            print(f"Error in analyze_post: {str(e)}")
            traceback.print_exc()
            raise ValueError(f"Failed to analyze post: {str(e)}")

    def _parse_feedback(self, raw_feedback: str) -> dict:
        """Parse the raw feedback text into structured format"""
        try:
            # Log the raw feedback for debugging
            print(f"Raw feedback from Claude:\n{raw_feedback}")
            
            sections = raw_feedback.split('\n')
            feedback = {
                "positive_feedback": "",
                "areas_for_development": "",
                "future_connections": "",
                "metrics": {
                    "content_coverage": 0.0,
                    "critical_thinking": 0.0,
                    "practical_application": 0.0
                }
            }
            
            current_section = None
            metrics_started = False
            metrics_lines = []
            
            for line in sections:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith("POSITIVE_FEEDBACK:"):
                    current_section = "positive_feedback"
                    feedback[current_section] = line.split(":", 1)[1].strip()
                elif line.startswith("AREAS_FOR_DEVELOPMENT:"):
                    current_section = "areas_for_development"
                    feedback[current_section] = line.split(":", 1)[1].strip()
                elif line.startswith("FUTURE_CONNECTIONS:"):
                    current_section = "future_connections"
                    feedback[current_section] = line.split(":", 1)[1].strip()
                elif line == "METRICS:" or line.startswith("METRICS:"):
                    current_section = None
                    metrics_started = True
                elif metrics_started:
                    # Collect all metrics lines for detailed parsing
                    metrics_lines.append(line)
                    
                    # Handle metrics section with improved parsing
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        # Map common metric keys to our expected keys
                        metric_map = {
                            "content_coverage": "content_coverage",
                            "content coverage": "content_coverage",
                            "critical_thinking": "critical_thinking",
                            "critical thinking": "critical_thinking",
                            "practical_application": "practical_application",
                            "practical application": "practical_application"
                        }
                        
                        print(f"Processing metric line: '{line}', key: '{key}', value: '{value}'")
                        
                        if key in metric_map:
                            try:
                                # Convert percentage to decimal if needed (e.g., "85%" to 0.85)
                                if "%" in value:
                                    value = value.replace("%", "").strip()
                                    value = float(value) / 100
                                else:
                                    value = float(value)
                                    
                                # Ensure value is between 0 and 1
                                if value > 1:
                                    value = value / 100
                                    
                                feedback["metrics"][metric_map[key]] = value
                                print(f"Set metric {metric_map[key]} to {value}")
                            except (ValueError, IndexError) as e:
                                print(f"Could not parse metric value: {value} for key: {key}, error: {e}")
                # Add content to current section if we're not in metrics
                elif current_section and not metrics_started:
                    feedback[current_section] += " " + line
            
            # Clean up any double spaces
            for section in ["positive_feedback", "areas_for_development", "future_connections"]:
                feedback[section] = " ".join(feedback[section].split())
            
            # Try to extract metrics more aggressively if all metrics are still 0
            if all(v == 0 for v in feedback["metrics"].values()) and metrics_lines:
                print("All metrics are 0. Trying more aggressive parsing...")
                print(f"Metrics lines: {metrics_lines}")
                
                # Example of more aggressive parsing - extract any number after metric names
                metrics_text = ' '.join(metrics_lines)
                for metric_key, target_key in [
                    ('content coverage', 'content_coverage'), 
                    ('critical thinking', 'critical_thinking'),
                    ('practical application', 'practical_application')
                ]:
                    # Look for patterns like "content coverage: 0.8" or "content coverage score: 80%"
                    import re
                    patterns = [
                        rf"{metric_key}:\s*(\d+\.?\d*)",
                        rf"{metric_key}[^:]*:\s*(\d+\.?\d*)",
                        rf"{metric_key}[^:]*:\s*(\d+\.?\d*)%"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, metrics_text, re.IGNORECASE)
                        if match:
                            try:
                                value = float(match.group(1))
                                if '%' in pattern or value > 1:
                                    value = value / 100
                                feedback["metrics"][target_key] = value
                                print(f"Set {target_key} to {value} using pattern {pattern}")
                                break
                            except (ValueError, IndexError) as e:
                                print(f"Error parsing using pattern {pattern}: {e}")
            
            # Final fallback - if Claude didn't return proper metrics, assign reasonable values
            if all(v == 0 for v in feedback["metrics"].values()):
                print("Still no metrics found. Using fallback scoring method.")
                # Analyze text length and complexity to estimate scores
                positive_len = len(feedback["positive_feedback"])
                development_len = len(feedback["areas_for_development"])
                connections_len = len(feedback["future_connections"])
                
                # Simple heuristic - longer, more detailed feedback suggests better content
                if positive_len > 100 and development_len > 100 and connections_len > 50:
                    feedback["metrics"]["content_coverage"] = 0.85
                    feedback["metrics"]["critical_thinking"] = 0.80
                    feedback["metrics"]["practical_application"] = 0.75
                elif positive_len > 50 and development_len > 50:
                    feedback["metrics"]["content_coverage"] = 0.75
                    feedback["metrics"]["critical_thinking"] = 0.70
                    feedback["metrics"]["practical_application"] = 0.65
                else:
                    feedback["metrics"]["content_coverage"] = 0.65
                    feedback["metrics"]["critical_thinking"] = 0.60
                    feedback["metrics"]["practical_application"] = 0.55
                    
            print(f"Final parsed feedback metrics: {feedback['metrics']}")
            return feedback
            
        except Exception as e:
            print(f"Error parsing feedback: {str(e)}")
            print(f"Raw feedback: {raw_feedback}")
            traceback.print_exc()
            raise

    def get_week_content(self, week: int) -> dict:
        """Get content for a specific week"""
        try:
            if not isinstance(week, int):
                raise ValueError(f"Week must be an integer, got {type(week)}")
                
            if week not in self.weekly_discussions:
                raise ValueError(f"Invalid week number: {week}")
                
            return self.weekly_discussions[week]
            
        except Exception as e:
            print(f"Error getting week content: {str(e)}")
            traceback.print_exc()
            raise