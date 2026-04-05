import pandas as pd

# Sample fertilizer data with environmental impact (approximate values)
FERTILIZER_DATA = {
    'Fertilizer Name': ['Urea', '14-35-14', '28-28', '10-26-26'],
    'Cost_per_kg': [0.5, 1.2, 1.0, 0.8],  # USD
    'CO2_emission_per_kg': [1.5, 2.0, 1.8, 1.3],  # kg CO2
    'Nitrogen_content': [46, 14, 28, 10],
    'Phosphorous_content': [0, 35, 28, 26],
    'Potassium_content': [0, 14, 0, 26]
}

def calculate_sustainability_score(fertilizer_name, quantity_needed):
    """
    Calculate sustainability metrics for a fertilizer recommendation.
    Returns cost, emissions, and efficiency score.
    """
    if fertilizer_name not in FERTILIZER_DATA['Fertilizer Name']:
        return {'cost': 0, 'emissions': 0, 'efficiency': 0}
    
    idx = FERTILIZER_DATA['Fertilizer Name'].index(fertilizer_name)
    
    cost = FERTILIZER_DATA['Cost_per_kg'][idx] * quantity_needed
    emissions = FERTILIZER_DATA['CO2_emission_per_kg'][idx] * quantity_needed
    
    # Efficiency based on nutrient content (higher is better)
    total_nutrients = (FERTILIZER_DATA['Nitrogen_content'][idx] + 
                      FERTILIZER_DATA['Phosphorous_content'][idx] + 
                      FERTILIZER_DATA['Potassium_content'][idx])
    efficiency = total_nutrients / (cost + emissions)  # Simplified metric
    
    return {
        'cost': round(cost, 2),
        'emissions': round(emissions, 2),
        'efficiency': round(efficiency, 2)
    }

def get_optimal_fertilizer(nitrogen_req, phosphorous_req, potassium_req):
    """
    Recommend fertilizer based on nutrient requirements with sustainability consideration.
    """
    best_fert = None
    best_score = float('inf')
    
    for fert in FERTILIZER_DATA['Fertilizer Name']:
        idx = FERTILIZER_DATA['Fertilizer Name'].index(fert)
        n_cont = FERTILIZER_DATA['Nitrogen_content'][idx]
        p_cont = FERTILIZER_DATA['Phosphorous_content'][idx]
        k_cont = FERTILIZER_DATA['Potassium_content'][idx]
        
        # Calculate quantity needed (simplified)
        qty = max(nitrogen_req / n_cont if n_cont > 0 else 0,
                 phosphorous_req / p_cont if p_cont > 0 else 0,
                 potassium_req / k_cont if k_cont > 0 else 0)
        
        metrics = calculate_sustainability_score(fert, qty)
        score = metrics['cost'] + metrics['emissions']  # Minimize cost + emissions
        
        if score < best_score:
            best_score = score
            best_fert = fert
    
    return best_fert

if __name__ == "__main__":
    # Example usage
    print(calculate_sustainability_score('Urea', 10))
    print(get_optimal_fertilizer(20, 10, 15))