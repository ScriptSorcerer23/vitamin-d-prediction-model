"""
Interactive Vitamin D Prediction Model Tester
Run this script to test the trained models with manual input.

Usage: python test_model_interactive.py
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import math
import os
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Custom XGBoost class used during training (needed for unpickling)
class XGBClassifierWithWeights(XGBClassifier):
    """Custom XGBoost classifier that automatically computes sample weights"""
    def fit(self, X, y, **kwargs):
        sample_weight = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def get_number_input(prompt, min_val=None, max_val=None, default=None):
    """Get validated numeric input from user"""
    while True:
        try:
            if default is not None:
                response = input(f"{prompt} (default: {default}): ").strip()
                if not response:
                    return default
                value = float(response)
            else:
                value = float(input(f"{prompt}: ").strip())
            
            if min_val is not None and value < min_val:
                print(f"❌ Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"❌ Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("❌ Please enter a valid number")

def get_choice_input(prompt, options):
    """Get validated choice input from user"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    
    while True:
        try:
            choice = int(input(f"Enter choice (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                return choice
            print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")

def get_yes_no(prompt):
    """Get yes/no input"""
    while True:
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'")

def collect_user_data():
    """Collect all required data from user"""
    data = {}
    
    print_header("VITAMIN D PREDICTION - QUESTIONNAIRE")
    print("Please answer the following questions to predict your vitamin D level.")
    print("This should take about 3-4 minutes.\n")
    
    # Section 1: Personal Information
    print_header("SECTION 1: Personal Information")
    
    data['Age'] = int(get_number_input("What is your age (years)", 1, 120))
    
    sex_choice = get_choice_input("What is your biological sex?", ["Male", "Female"])
    data['Sex'] = sex_choice  # 1=Male, 2=Female
    
    ethnicity_choice = get_choice_input(
        "Which ethnicity best describes you?",
        [
            "Mexican American",
            "Other Hispanic",
            "Non-Hispanic White",
            "Non-Hispanic Black",
            "Other/Mixed"
        ]
    )
    data['Ethnicity'] = ethnicity_choice
    
    # Section 2: Body Measurements
    print_header("SECTION 2: Body Measurements")
    
    use_metric = get_yes_no("Use metric units (kg/cm)?")
    
    if use_metric:
        height_cm = get_number_input("Height (cm)", 50, 250)
        weight_kg = get_number_input("Weight (kg)", 20, 300)
    else:
        feet = int(get_number_input("Height - Feet", 1, 8))
        inches = get_number_input("Height - Inches", 0, 11)
        height_cm = (feet * 30.48) + (inches * 2.54)
        
        weight_lbs = get_number_input("Weight (lbs)", 44, 660)
        weight_kg = weight_lbs * 0.453592
    
    height_m = height_cm / 100
    data['BMI'] = weight_kg / (height_m ** 2)
    print(f"✓ Your BMI: {data['BMI']:.1f}")
    
    # Section 3: Dietary Habits
    print_header("SECTION 3: Dietary Habits")
    
    takes_supplements = get_yes_no("Do you take vitamin D supplements?")
    if takes_supplements:
        supplement_iu = get_number_input("How much vitamin D supplement (IU/day)", 0, 10000, 1000)
        data['supplement_vitd_mcg'] = supplement_iu / 40  # Convert IU to mcg
    else:
        data['supplement_vitd_mcg'] = 0
    
    print("\nHow often do you eat vitamin D-rich foods?")
    print("(Examples: fatty fish, fortified milk, egg yolks)")
    food_freq = get_choice_input(
        "Select frequency:",
        [
            "Rarely (0-1 servings/week)",
            "Sometimes (2-4 servings/week)",
            "Often (5-7 servings/week)",
            "Daily (8+ servings/week)"
        ]
    )
    dietary_map = {1: 2, 2: 5, 3: 8, 4: 12}
    data['dietary_vitd_mcg'] = dietary_map[food_freq]
    
    calorie_choice = get_choice_input(
        "Estimate your average daily calorie intake:",
        [
            "Low (1200-1600 kcal) - Small meals, weight loss",
            "Moderate (1600-2200 kcal) - Average adult",
            "High (2200-2800 kcal) - Active lifestyle",
            "Very High (2800+ kcal) - Athletes, bodybuilders"
        ]
    )
    calorie_map = {1: 1400, 2: 1900, 3: 2500, 4: 3200}
    data['calories'] = calorie_map[calorie_choice]
    
    # Section 4: Lifestyle & Activity
    print_header("SECTION 4: Lifestyle & Activity")
    
    smoked = get_yes_no("Have you smoked at least 100 cigarettes in your life?")
    data['smoked_100_cigarettes'] = 1 if smoked else 2
    
    print("\nDoes your work involve vigorous physical activity?")
    print("(Examples: construction, heavy lifting, fast cycling)")
    vigorous = get_yes_no("Vigorous work activity?")
    data['vigorous_work'] = 1 if vigorous else 2
    
    print("\nDoes your work involve moderate physical activity?")
    print("(Examples: walking, light lifting, standing)")
    moderate = get_yes_no("Moderate work activity?")
    data['moderate_work'] = 1 if moderate else 2
    
    # Section 5: Optional - Blood Pressure & Medications
    print_header("SECTION 5: Health Information (Optional)")
    print("The following questions are optional but improve accuracy.")
    print("Press Enter to skip and use default values.\n")
    
    data['systolic_bp'] = get_number_input("Systolic blood pressure (mmHg)", 80, 200, 120)
    data['diastolic_bp'] = get_number_input("Diastolic blood pressure (mmHg)", 40, 120, 80)
    
    print("\nDo you take any of the following medications?")
    data['takes_glucocorticoid'] = 1 if get_yes_no("Glucocorticoids (e.g., prednisone)?") else 0
    data['takes_anticonvulsant'] = 1 if get_yes_no("Anticonvulsants (e.g., phenytoin)?") else 0
    data['takes_cholesterol_binder'] = 1 if get_yes_no("Cholesterol binders (e.g., cholestyramine)?") else 0
    data['takes_weightloss_drug'] = 1 if get_yes_no("Weight loss drugs (e.g., orlistat)?") else 0
    data['takes_vitd_affecting_med'] = 1 if get_yes_no("Any other vitamin D-affecting medication?") else 0
    
    return data

def engineer_features(data):
    """Calculate engineered features"""
    # Auto-detect current month
    data['exam_month'] = datetime.now().month
    
    # Seasonal encoding
    data['month_sin'] = math.sin(2 * math.pi * data['exam_month'] / 12)
    data['month_cos'] = math.cos(2 * math.pi * data['exam_month'] / 12)
    
    # Polynomial features
    data['BMI_squared'] = data['BMI'] ** 2
    data['Age_squared'] = data['Age'] ** 2
    
    # Total vitamin D intake
    data['total_vitd_intake'] = data['supplement_vitd_mcg'] + data['dietary_vitd_mcg']
    
    # Interaction terms
    data['Ethnicity_BMI'] = data['Ethnicity'] * data['BMI']
    data['supplement_season_sin'] = data['supplement_vitd_mcg'] * data['month_sin']
    data['supplement_season_cos'] = data['supplement_vitd_mcg'] * data['month_cos']
    
    # BMI category
    bmi = data['BMI']
    if bmi < 18.5:
        data['BMI_category'] = 0
    elif bmi < 25:
        data['BMI_category'] = 1
    elif bmi < 30:
        data['BMI_category'] = 2
    else:
        data['BMI_category'] = 3
    
    return data

def load_models():
    """Load trained models and scaler"""
    try:
        print_header("Loading Models...")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models')
        
        print(f"Looking for models in: {models_dir}")
        
        scaler = joblib.load(os.path.join(models_dir, 'scaler_optimized.pkl'))
        print("✓ Scaler loaded")
        
        regression_model = joblib.load(os.path.join(models_dir, 'lightgbm_optimized_vitamin_d.pkl'))
        print("✓ Regression model loaded")
        
        classification_model = joblib.load(os.path.join(models_dir, 'xgboost_classifier_optimized_vitamin_d.pkl'))
        print("✓ Classification model loaded")
        
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names_optimized.pkl'))
        print(f"✓ Feature names loaded ({len(feature_names)} features)")
        
        return scaler, regression_model, classification_model, feature_names
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Model files not found!")
        print(f"   {e}")
        print("\nPlease ensure the models have been saved by running:")
        print("   Cell 62 in ml_model/notebooks/07_optimized_model.ipynb")
        print(f"\nExpected location: {os.path.join(script_dir, 'models')}")
        return None, None, None, None

def make_predictions(data, scaler, regression_model, classification_model, feature_names):
    """Make predictions using the models"""
    # Create feature vector in correct order
    feature_vector = [data[name] for name in feature_names]
    X = np.array([feature_vector])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Regression prediction
    predicted_level = regression_model.predict(X_scaled)[0]
    
    # Classification prediction
    predicted_class = classification_model.predict(X_scaled)[0]
    class_probs = classification_model.predict_proba(X_scaled)[0]
    
    categories = ['Deficient (<20 ng/mL)', 'Insufficient (20-30 ng/mL)', 'Sufficient (≥30 ng/mL)']
    
    return {
        'level': predicted_level,
        'category': categories[predicted_class],
        'category_code': predicted_class,
        'probabilities': {
            'Deficient': class_probs[0],
            'Insufficient': class_probs[1],
            'Sufficient': class_probs[2]
        }
    }

def display_results(data, prediction):
    """Display prediction results in a nice format"""
    print_header("VITAMIN D PREDICTION RESULTS")
    
    # User summary
    print("📋 Your Profile:")
    print(f"   Age: {data['Age']} years")
    print(f"   Sex: {'Male' if data['Sex'] == 1 else 'Female'}")
    print(f"   BMI: {data['BMI']:.1f}")
    print(f"   Vitamin D Supplement: {data['supplement_vitd_mcg']:.1f} mcg/day")
    print(f"   Dietary Vitamin D: {data['dietary_vitd_mcg']:.1f} mcg/day")
    
    # Regression result
    print(f"\n🔬 PREDICTED VITAMIN D LEVEL:")
    print(f"   {prediction['level']:.1f} ng/mL")
    print(f"   (95% confidence: {prediction['level']-8.57:.1f} - {prediction['level']+8.57:.1f} ng/mL)")
    
    # Classification result
    print(f"\n📊 VITAMIN D STATUS:")
    print(f"   {prediction['category']}")
    print(f"\n   Probability breakdown:")
    for category, prob in prediction['probabilities'].items():
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"   {category:25s} [{bar}] {prob*100:.1f}%")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    category_code = prediction['category_code']
    
    if category_code == 0:  # Deficient
        print("   ⚠️  Your vitamin D level appears DEFICIENT.")
        print("\n   Recommendations:")
        print("   • Consult your doctor for high-dose vitamin D therapy")
        print("   • Consider taking 2000-4000 IU vitamin D supplement daily")
        print("   • Increase sun exposure (15-30 minutes/day)")
        print("   • Get a blood test to confirm and monitor levels")
    
    elif category_code == 1:  # Insufficient
        print("   ⚡ Your vitamin D level appears INSUFFICIENT.")
        print("\n   Recommendations:")
        print("   • Take 1000-2000 IU vitamin D supplement daily")
        print("   • Increase sun exposure (15-20 minutes/day)")
        print("   • Eat more vitamin D-rich foods (fatty fish, fortified milk)")
        print("   • Recheck level in 3 months")
    
    else:  # Sufficient
        print("   ✅ Your vitamin D level appears SUFFICIENT. Great job!")
        print("\n   Recommendations:")
        print("   • Maintain current sun exposure and diet")
        print("   • Continue taking supplements if currently using")
        print("   • Monitor during winter months")
        print("   • Recheck level annually")
    
    print("\n⚠️  DISCLAIMER:")
    print("   This is a prediction model, not a medical diagnosis.")
    print("   For accurate vitamin D levels, get a blood test (25-OH vitamin D).")
    print("   Consult your healthcare provider for medical advice.")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("SOLMATE VITAMIN D PREDICTION MODEL - INTERACTIVE TESTER")
    print("="*60)
    
    # Load models
    scaler, regression_model, classification_model, feature_names = load_models()
    
    if scaler is None:
        return
    
    # Collect data
    data = collect_user_data()
    
    # Engineer features
    print_header("Processing Your Data...")
    data = engineer_features(data)
    print("✓ Features calculated")
    print(f"✓ Current month: {datetime.now().strftime('%B')} (seasonal factor included)")
    
    # Make predictions
    print("✓ Scaling features...")
    print("✓ Running models...")
    prediction = make_predictions(data, scaler, regression_model, classification_model, feature_names)
    
    # Display results
    display_results(data, prediction)
    
    # Ask if user wants to try again
    if get_yes_no("Would you like to test with different inputs?"):
        print("\n\n")
        main()
    else:
        print("Thank you for using the Vitamin D Prediction Model! 👋")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Test cancelled by user.")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
