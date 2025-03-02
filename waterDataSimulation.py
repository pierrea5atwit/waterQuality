import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

"""
_summary_

The training & usage of A.I. classification models in order to recognize anomalies and/or unsafe metrics with respect to water quality


_citation_
Source: Thorslund, Josefin; van Vliet, Michelle T H (2020): A global salinity dataset of surface water and 
groundwater measurements from 1980-2019 [dataset]. PANGAEA, https://doi.org/10.1594/PANGAEA.913939
"""

# Define safe ranges for crop irrigation
SAFE_RANGES = {
    "pH": (6.0, 8.0),  # Ideal pH range for most crops
    "Turbidity": (0, 5),  # NTU, clearer water is better
    "TDS": (0, 500)  # ppm, lower is generally better
}

# Converts binary classification to a 0-100 safety score
def quality_to_safety_score(quality_label, confidence, features, safe_ranges=SAFE_RANGES):
    # Base score from model prediction (0-70 points)
    base_score = 70 if quality_label == 0 else 30
    
    # Scale by confidence (0-20 points)
    confidence_points = confidence * 20
    
    # Parameter deviation penalty (0-10 points)
    parameter_score = 10
    deviations = []
    
    for param, value in features.items():
        if param in safe_ranges:
            min_val, max_val = safe_ranges[param]
            if value < min_val:
                deviation = (min_val - value) / min_val
                deviations.append((param, deviation, "low"))
                parameter_score -= min(deviation * 5, 3.33)
            elif value > max_val:
                deviation = (value - max_val) / max_val
                deviations.append((param, deviation, "high"))
                parameter_score -= min(deviation * 5, 3.33)
    
    # Find biggest concern
    biggest_concern = None
    if deviations:
        biggest_concern = max(deviations, key=lambda x: x[1])
    
    final_score = min(max(base_score + confidence_points + parameter_score, 0), 100)
    return round(final_score, 1), biggest_concern

# Denotes quality as 'good' or 'bad' based on specific parameters
def label_quality(pH, turbidity, TDS):
    if (SAFE_RANGES["pH"][0] <= pH <= SAFE_RANGES["pH"][1] and 
        turbidity < SAFE_RANGES["Turbidity"][1] and 
        TDS <= SAFE_RANGES["TDS"][1]):
        return 0  # good
    else:
        return 1  # bad

# Simulate water quality sensor readings
def generate_sensor_data(num_samples=100, is_test_set=False):
    data_list = []
    for _ in range(num_samples):
        # For test set, we can optionally introduce some distribution shift
        if is_test_set:
            # Slightly different distribution for test data
            data_list.append({
                "pH": round(random.uniform(5.0, 9.0), 2),
                "Turbidity": round(random.uniform(0, 15), 2),
                "TDS": round(random.uniform(50, 1200), 2),
            })
        else:
            data_list.append({
                "pH": round(random.uniform(5.5, 8.5), 2),
                "Turbidity": round(random.uniform(0, 10), 2),
                "TDS": round(random.uniform(100, 1000), 2),
            })
    return pd.DataFrame(data_list)

# Train and save model
def train_and_save_model(output_dir="models"):
    # Generate training dataset
    train_data = generate_sensor_data(num_samples=500)
    train_data['quality'] = train_data.apply(lambda row: label_quality(row['pH'], row['Turbidity'], row['TDS']), axis=1)

    # Features and labels
    X = train_data[['pH', 'Turbidity', 'TDS']]
    y = train_data['quality']

    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert scaled arrays back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Validation accuracy
    y_val_pred = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Model Validation Accuracy: {val_accuracy * 100:.2f}%\n")
    
    # Create directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model and scaler
    joblib.dump(model, os.path.join(output_dir, 'irrigation_water_quality_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    print(f"Model and scaler saved to {output_dir} directory\n")
    
    return model, scaler

# Load saved model
def load_model(model_dir="models"):
    model_path = os.path.join(model_dir, 'irrigation_water_quality_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Training a new model...\n")
        return train_and_save_model(model_dir)
    
    print("Loading saved model and scaler...\n")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Predict single water sample
def predict_water_quality(model, scaler, sample):
    # Make copy to avoid modifying the original
    sample_df = pd.DataFrame(sample)
    
    # Scale the input
    sample_scaled = scaler.transform(sample_df)
    sample_scaled = pd.DataFrame(sample_scaled, columns=sample_df.columns)
    
    # Get prediction and confidence
    prediction = model.predict(sample_scaled)[0]
    confidence = model.predict_proba(sample_scaled)[0].max()
    
    # Calculate safety score (0-100)
    safety_score, biggest_concern = quality_to_safety_score(
        prediction, confidence, {k: v[0] for k, v in sample.items()}
    )
    
    # Format output
    quality_text = "Good" if prediction == 0 else "Concerning"
    
    print("\n" + "="*50)
    print(" WATER QUALITY ANALYSIS FOR CROP IRRIGATION ")
    print("="*50)
    
    # Print sample values
    for col in sample:
        print(f"{col:>10}: {sample[col][0]:.2f}")
    
    print("-"*50)
    print(f"Quality Assessment: {quality_text} (Confidence: {confidence*100:.1f}%)")
    print(f"Safety Score: {safety_score}/100")
    
    # Print biggest concern if any
    if biggest_concern:
        param, deviation, direction = biggest_concern
        print(f"Biggest Concern: {param} is too {direction}")
        print(f"Recommended Range: {SAFE_RANGES[param][0]} - {SAFE_RANGES[param][1]}")
    else:
        print("All parameters within acceptable ranges")
    
    print("="*50)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "safety_score": safety_score,
        "biggest_concern": biggest_concern
    }

# Visualize water quality parameters
def visualize_water_quality(sample, result):
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create 2x2 subplot layout
    plt.subplot(2, 2, 1)
    
    # 1. Parameter ranges visualization
    params = list(SAFE_RANGES.keys())
    values = [sample[param][0] for param in params]
    colors = []
    
    for i, param in enumerate(params):
        min_val, max_val = SAFE_RANGES[param]
        if min_val <= values[i] <= max_val:
            colors.append('green')
        else:
            colors.append('red')
    
    # Create horizontal bar chart
    bars = plt.barh(params, values, color=colors, alpha=0.7)
    
    # Add safe range markers
    for i, param in enumerate(params):
        min_val, max_val = SAFE_RANGES[param]
        plt.axvline(x=min_val, ymin=i/len(params), ymax=(i+1)/len(params), 
                    color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=max_val, ymin=i/len(params), ymax=(i+1)/len(params), 
                    color='black', linestyle='--', alpha=0.5)
       
    
    plt.title("Water Parameters vs Safe Ranges")
    plt.xlabel("Value")
    plt.ylabel("Parameter")
    
    # 2. Safety score gauge
    plt.subplot(2, 2, 2)
    safety = result['safety_score']
    
    # Create gauge chart using pie chart
    size = 0.3
    plt.pie([safety, 100-safety], 
            colors=['#2196F3', '#ECEFF1'], 
            startangle=90, 
            counterclock=False, 
            radius=1,
            wedgeprops={"width":size, "edgecolor":"w"})
    
    plt.text(0, 0, f"{safety}/100", ha='center', va='center', fontsize=24)
    plt.title("Irrigation Safety Score")
    
    # 3. Confidence visualization
    plt.subplot(2, 2, 3)
    confidence = result['confidence'] * 100
    
    # Create confidence bar
    plt.bar(['Confidence'], [confidence], color='#4CAF50', alpha=0.7)
    plt.ylim(0, 100)
    plt.title("Model Confidence")
    plt.ylabel("Confidence (%)")
    
    # 4. Recommendation Box
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    if result['prediction'] == 0:
        recommendation = "✓ This water source is suitable for crop irrigation."
        if result['biggest_concern']:
            param, _, direction = result['biggest_concern']
            recommendation += f"\n\nMonitor {param} levels as they are\nclose to the {direction} end of the acceptable range."
    else:
        if result['biggest_concern']:
            param, _, direction = result['biggest_concern']
            recommendation = f"⚠ Water quality concerns detected.\n\nPrimary issue: {param} is too {direction}.\n"
            
            if direction == "high":
                recommendation += f"Consider dilution or treatment\nto reduce {param} before use."
            else:
                recommendation += f"Consider supplementing to\nincrease {param} before use."
        else:
            recommendation = "⚠ Multiple marginal parameters detected.\nRecommend additional testing and treatment."
    
    plt.text(0.5, 0.5, recommendation, ha='center', va='center', fontsize=12)
    plt.title("Recommendation")
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load or train model
    model, scaler = load_model()
    
    # Generate a sample or use provided data
    sample = {"pH": [7.5], "Turbidity": [3.0], "TDS": [450]}
    
    # Optional: Uncomment to use random sample
    random_data = generate_sensor_data(num_samples=1)
    sample = {col: random_data[col].values for col in random_data.columns}
    
    # Predict water quality
    result = predict_water_quality(model, scaler, sample)
    
    # Visualize results
    visualize_water_quality(sample, result)

# Run the program
if __name__ == "__main__":
    main()