# forecasting.py
from flask import jsonify, request
import numpy as np
from scipy import stats
import traceback

class RevenueForecast:
    def __init__(self):
        # Scale factors for confidence score impact (0-10 scale)
        self.confidence_coefficients = {
            'variance_factor': 0.05,  # 5% per confidence point difference from 10
            'best_case_multiplier': 0.02,  # 2% increase per confidence point
            'worst_case_multiplier': 0.03   # 3% decrease per confidence point
        }

    def calculate_variance_ranges(self, base_value, confidence_score):
        """
        Calculate forecast ranges based on confidence score (0-10 scale)
        Higher confidence = narrower range between best/worst case
        """
        try:
            confidence_score = float(confidence_score)
            if not 0 <= confidence_score <= 10:
                raise ValueError("Confidence score must be between 0 and 10")
                
            # Calculate variance based on confidence score
            # Lower confidence = higher variance
            variance_multiplier = (10 - confidence_score) * self.confidence_coefficients['variance_factor']
            
            # Calculate case multipliers
            best_case_boost = confidence_score * self.confidence_coefficients['best_case_multiplier']
            worst_case_reduction = (10 - confidence_score) * self.confidence_coefficients['worst_case_multiplier']
            
            # Calculate ranges with upper and lower bounds
            best_case = base_value * (1 + best_case_boost + variance_multiplier)
            worst_case = base_value * (1 - worst_case_reduction - variance_multiplier)
            expected_case = base_value
            
            # Calculate error margin and confidence interval
            error_margin = base_value * variance_multiplier
            confidence_interval = confidence_score * 10  # Scale to percentage
            
            return {
                'best_case': best_case,
                'expected_case': expected_case,
                'worst_case': worst_case,
                'error_margin': error_margin,
                'confidence_interval': confidence_interval
            }
            
        except ValueError as e:
            raise ValueError(f"Invalid confidence score: {str(e)}")

    def handle_upsell_forecast(self):
        try:
            data = request.get_json()
            
            scenarios = data.get('scenarios', {})
            if not scenarios:
                return jsonify({"error": "No scenario data provided"}), 400

            best = scenarios.get('best', {})
            worst = scenarios.get('worst', {})
            expected = scenarios.get('expected', {})
            confidence_level = float(data.get('confidence_level', 5))

            # Calculate revenue for each scenario
            def calculate_revenue(scenario):
                customers = float(scenario.get('customers', 0))
                rate = float(scenario.get('rate', 0)) / 100  # Convert percentage to decimal
                price = float(scenario.get('price', 0))
                cost = float(scenario.get('cost', 0))
                
                gross_revenue = customers * rate * price
                return gross_revenue - cost

            # Calculate each case
            best_case = calculate_revenue(best)
            worst_case = calculate_revenue(worst)
            expected_case = calculate_revenue(expected)

            # Calculate additional metrics using expected case
            gross_revenue = expected['customers'] * (expected['rate'] / 100) * expected['price']
            potential_upgrades = expected['customers'] * (expected['rate'] / 100)

            result = {
                'best_case': float(best_case),
                'worst_case': float(worst_case),
                'expected_case': float(expected_case),
                'confidence_level': confidence_level,
                'error_margin': abs(best_case - worst_case) / 2,
                'metrics': {
                    'potential_upgrades': float(potential_upgrades),
                    'gross_revenue': float(gross_revenue),
                    'implementation_cost': float(expected['cost']),
                    'upgrade_rate': float(expected['rate']),
                    'revenue_per_customer': float(expected['price'])
                }
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error in upsell forecast: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def handle_internal_forecast(self):
        try:
            data = request.get_json()
            scenarios = data.get('scenarios', {})
            if not scenarios:
                return jsonify({"error": "No scenario data provided"}), 400

            best = scenarios.get('best', {})
            worst = scenarios.get('worst', {})
            expected = scenarios.get('expected', {})
            confidence_level = float(data.get('confidence_level', 5))

            def calculate_savings(scenario):
                annual_hours = float(scenario.get('annual_hours', 0))
                avg_salary = float(scenario.get('avg_salary', 0))
                velocity_increase = float(scenario.get('velocity_increase', 0)) / 100
                tool_cost = float(scenario.get('tool_cost', 0))
                
                current_cost = annual_hours * avg_salary
                efficiency_gain = (1 - (1 / (1 + velocity_increase)))
                cost_savings = current_cost * efficiency_gain
                return cost_savings - tool_cost

            best_case = calculate_savings(best)
            worst_case = calculate_savings(worst)
            expected_case = calculate_savings(expected)

            # Calculate metrics using expected case
            current_cost = expected['annual_hours'] * expected['avg_salary']
            efficiency_gain = (1 - (1 / (1 + expected['velocity_increase'] / 100)))

            result = {
                'best_case': float(best_case),
                'worst_case': float(worst_case),
                'expected_case': float(expected_case),
                'confidence_level': confidence_level,
                'error_margin': abs(best_case - worst_case) / 2,
                'metrics': {
                    'current_annual_cost': float(current_cost),
                    'tool_cost': float(expected['tool_cost']),
                    'efficiency_gain_percent': float(efficiency_gain * 100),
                    'annual_savings': float(current_cost * efficiency_gain),
                    'velocity_improvement': float(expected['velocity_increase']),
                    'hours_saved': float(expected['annual_hours'] * efficiency_gain)
                }
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error in internal forecast: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def handle_market_forecast(self):
        try:
            data = request.get_json()
            scenarios = data.get('scenarios', {})
            if not scenarios:
                return jsonify({"error": "No scenario data provided"}), 400

            best = scenarios.get('best', {})
            worst = scenarios.get('worst', {})
            expected = scenarios.get('expected', {})
            confidence_level = float(data.get('confidence_level', 5))

            def calculate_revenue(scenario):
                total_market = float(scenario.get('customers', 0))
                penetration = float(scenario.get('rate', 0)) / 100
                contract_value = float(scenario.get('value', 0))
                costs = float(scenario.get('costs', 0))
                competition = float(scenario.get('competition', 0)) / 100
                
                available_market = total_market * (1 - competition)
                potential_customers = available_market * penetration
                gross_revenue = potential_customers * contract_value
                return gross_revenue - costs

            best_case = calculate_revenue(best)
            worst_case = calculate_revenue(worst)
            expected_case = calculate_revenue(expected)

            # Calculate metrics using expected case
            available_market = expected['customers'] * (1 - expected['competition'] / 100)
            potential_customers = available_market * (expected['rate'] / 100)
            gross_revenue = potential_customers * expected['value']

            result = {
                'best_case': float(best_case),
                'worst_case': float(worst_case),
                'expected_case': float(expected_case),
                'confidence_level': confidence_level,
                'error_margin': abs(best_case - worst_case) / 2,
                'metrics': {
                    'available_market': float(available_market),
                    'potential_customers': float(potential_customers),
                    'gross_revenue': float(gross_revenue),
                    'initial_costs': float(expected['costs']),
                    'effective_penetration': float(expected['rate'] * (1 - expected['competition'] / 100)),
                    'market_share_remaining': float(100 - expected['competition'])
                }
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error in market forecast: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def handle_satisfaction_forecast(self):
        try:
            data = request.get_json()
            scenarios = data.get('scenarios', {})
            if not scenarios:
                return jsonify({"error": "No scenario data provided"}), 400

            best = scenarios.get('best', {})
            worst = scenarios.get('worst', {})
            expected = scenarios.get('expected', {})
            confidence_level = float(data.get('confidence_level', 5))

            def calculate_impact(scenario):
                current_nps = float(scenario.get('current_nps', 0))
                target_nps = float(scenario.get('target_nps', 0))
                current_churn = float(scenario.get('current_churn', 0)) / 100
                initiative_cost = float(scenario.get('initiative_cost', 0))
                revenue_impact = float(scenario.get('revenue_impact', 0))
                
                nps_improvement = target_nps - current_nps
                direct_impact = nps_improvement * revenue_impact
                
                churn_reduction = (nps_improvement / 10) * 0.2
                retention_impact = current_churn * churn_reduction * direct_impact
                
                total_impact = direct_impact + retention_impact
                return total_impact - initiative_cost

            best_case = calculate_impact(best)
            worst_case = calculate_impact(worst)
            expected_case = calculate_impact(expected)

            # Calculate metrics using expected case
            nps_improvement = expected['target_nps'] - expected['current_nps']
            direct_impact = nps_improvement * expected['revenue_impact']
            churn_reduction = (nps_improvement / 10) * 0.2
            retention_impact = (expected['current_churn'] / 100) * churn_reduction * direct_impact

            result = {
                'best_case': float(best_case),
                'worst_case': float(worst_case),
                'expected_case': float(expected_case),
                'confidence_level': confidence_level,
                'error_margin': abs(best_case - worst_case) / 2,
                'metrics': {
                    'nps_improvement': float(nps_improvement),
                    'direct_revenue_impact': float(direct_impact),
                    'churn_improvement_percentage': float(churn_reduction * 100),
                    'retention_revenue': float(retention_impact),
                    'total_revenue_impact': float(direct_impact + retention_impact),
                    'initiative_cost': float(expected['initiative_cost'])
                }
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error in satisfaction forecast: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500