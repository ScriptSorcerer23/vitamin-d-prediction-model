"""
Gradio Interface for Vitamin D Level Prediction Model
A modern, interactive web interface for predicting vitamin D levels based on user inputs.

Usage: python gradio_vitamin_d_app.py
"""

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os
import json
import math
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Custom XGBoost class used during training (needed for unpickling)
class XGBClassifierWithWeights(XGBClassifier):
    """Custom XGBoost classifier that automatically computes sample weights"""
    def fit(self, X, y, **kwargs):
        sample_weight = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)

# ==================== MODEL LOADING ====================

def load_models():
    """Load trained models and scaler"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    try:
        scaler = joblib.load(os.path.join(models_dir, 'scaler_optimized.pkl'))
        regression_model = joblib.load(os.path.join(models_dir, 'lightgbm_optimized_vitamin_d.pkl'))
        classification_model = joblib.load(os.path.join(models_dir, 'xgboost_classifier_optimized_vitamin_d.pkl'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names_optimized.pkl'))
        
        with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return scaler, regression_model, classification_model, feature_names, metadata
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load models at startup
try:
    scaler, regression_model, classification_model, feature_names, metadata = load_models()
except Exception as e:
    print(f"Warning: Could not load models at startup: {e}")
    scaler = regression_model = classification_model = feature_names = metadata = None

# ==================== FEATURE ENGINEERING ====================

def engineer_features(user_data):
    """Calculate engineered features from user input"""
    # Current month for seasonal encoding
    exam_month = datetime.now().month
    
    # Seasonal encoding (sine-cosine for cyclical nature of seasons)
    month_sin = math.sin(2 * math.pi * exam_month / 12)
    month_cos = math.cos(2 * math.pi * exam_month / 12)
    
    # Polynomial features
    bmi_squared = user_data['BMI'] ** 2
    age_squared = user_data['Age'] ** 2
    
    # Total vitamin D intake
    total_vitd_intake = user_data['supplement_vitd_mcg'] + user_data['dietary_vitd_mcg']
    
    # Interaction terms
    ethnicity_bmi = user_data['Ethnicity'] * user_data['BMI']
    supplement_season_sin = user_data['supplement_vitd_mcg'] * month_sin
    supplement_season_cos = user_data['supplement_vitd_mcg'] * month_cos
    
    # BMI category
    bmi = user_data['BMI']
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3
    
    return {
        'month_sin': month_sin,
        'month_cos': month_cos,
        'BMI_squared': bmi_squared,
        'Age_squared': age_squared,
        'total_vitd_intake': total_vitd_intake,
        'Ethnicity_BMI': ethnicity_bmi,
        'supplement_season_sin': supplement_season_sin,
        'supplement_season_cos': supplement_season_cos,
        'BMI_category': bmi_category
    }

def prepare_features(user_data, engineered_features):
    """Prepare features in the correct order for the model"""
    feature_vector = []
    
    for feature in feature_names:
        if feature in user_data:
            feature_vector.append(user_data[feature])
        elif feature in engineered_features:
            feature_vector.append(engineered_features[feature])
    
    return np.array(feature_vector).reshape(1, -1)

# ==================== PREDICTION & CLASSIFICATION ====================

def get_vitamin_d_status(vit_d_level):
    """Determine vitamin D status and color coding"""
    if vit_d_level < 20:
        status = "🔴 Deficient"
        status_text = "Deficient"
        color = "#EF4444"  # Red
        interpretation = "Your vitamin D level is dangerously low. You should consult with a healthcare provider about vitamin D supplementation and increase sun exposure when possible."
    elif vit_d_level < 30:
        status = "🟠 Insufficient"
        status_text = "Insufficient"
        color = "#F97316"  # Orange
        interpretation = "Your vitamin D level is below the recommended range. Consider increasing vitamin D intake through supplements, fortified foods, or more sun exposure."
    else:
        status = "🟢 Sufficient"
        status_text = "Sufficient"
        color = "#22C55E"  # Green
        interpretation = "Your vitamin D level is adequate. Maintain your current vitamin D intake and sun exposure habits."
    
    return status, status_text, color, interpretation

def create_gauge_chart(vit_d_level):
    """Create a beautiful gauge chart for vitamin D level"""
    status, status_text, color, _ = get_vitamin_d_status(vit_d_level)
    
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=vit_d_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Vitamin D Level (ng/mL)", 'font': {'size': 24}},
        delta={'reference': 30, 'suffix': " from recommended"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#FEE2E2"},
                {'range': [20, 30], 'color': "#FFEDD5"},
                {'range': [30, 100], 'color': "#DCFCE7"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 20
            }
        },
        number={'suffix': " ng/mL", 'font': {'size': 32, 'color': color}}
    )])
    
    fig.update_layout(
        font={'family': "Arial, sans-serif", 'size': 14},
        height=450,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.9)'
    )
    
    return fig

def create_comparison_chart(vit_d_level):
    """Create a horizontal bar chart showing vitamin D ranges and user's level"""
    status, status_text, color, _ = get_vitamin_d_status(vit_d_level)
    
    # Define ranges with colors
    ranges = [
        {'name': 'Deficient', 'min': 0, 'max': 20, 'color': '#EF4444'},
        {'name': 'Insufficient', 'min': 20, 'max': 30, 'color': '#F97316'},
        {'name': 'Sufficient', 'min': 30, 'max': 100, 'color': '#22C55E'}
    ]
    
    fig = go.Figure()
    
    # Add background ranges
    for i, r in enumerate(ranges):
        fig.add_trace(go.Bar(
            y=['Vitamin D Level'],
            x=[r['max'] - r['min']],
            name=r['name'],
            orientation='h',
            marker=dict(color=r['color']),
            base=r['min'],
            hovertemplate=f"<b>{r['name']}</b><br>{r['min']}-{r['max']} ng/mL<extra></extra>",
            showlegend=True
        ))
    
    # Add user's level as a line
    fig.add_vline(
        x=vit_d_level,
        line_dash="dash",
        line_color=color,
        line_width=3
    )
    
    # Add recommended target line
    fig.add_vline(
        x=30,
        line_dash="dot",
        line_color="gray",
        line_width=2
    )
    
    # Add user's level annotation with arrow
    fig.add_annotation(
        x=vit_d_level,
        y=0.5,
        text=f"Your Level:<br>{vit_d_level:.1f} ng/mL",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=color,
        font=dict(size=10, color=color),
        xanchor='center',
        ax=0,
        ay=-40
    )
    
    fig.update_layout(
        barmode='stack',
        xaxis_title='Vitamin D Level (ng/mL)',
        yaxis_title='',
        height=250,
        margin=dict(l=50, r=200, t=100, b=30),
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(255,255,255,0.95)',
        xaxis=dict(range=[0, 80]),
        hovermode='closest',
        font={'family': "Arial, sans-serif", 'size': 12},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def predict_vitamin_d(age, sex, ethnicity, height_cm, weight_kg, 
                     takes_supplement, supplement_vitd_iu, dietary_freq, calorie_intake,
                     smoking_history, vigorous_activity, moderate_activity,
                     systolic_bp, diastolic_bp, 
                     glucocorticoid, anticonvulsant, cholesterol_binder, 
                     weightloss_drug, vitd_med):
    """
    Make vitamin D prediction based on user inputs
    """
    # Check if models are loaded
    if regression_model is None or classification_model is None or scaler is None:
        return (
            gr.update(visible=True),
            None,
            None,
            "❌ Models not loaded. Please ensure all model files are in the models/ directory.",
            "Error"
        )
    
    try:
        # Convert dietary frequency to mcg values (matching test_model_interactive.py)
        dietary_map = {
            "Rarely (0-1 servings/week)": 2,
            "Sometimes (2-4 servings/week)": 5,
            "Often (5-7 servings/week)": 8,
            "Daily (8+ servings/week)": 12
        }
        dietary_vitd_mcg = dietary_map.get(dietary_freq, 5)
        
        # Convert calorie category to value (matching test_model_interactive.py)
        calorie_map = {
            "Low (1200-1600 kcal) - Small meals, weight loss": 1400,
            "Moderate (1600-2200 kcal) - Average adult": 1900,
            "High (2200-2800 kcal) - Active lifestyle": 2500,
            "Very High (2800+ kcal) - Athletes, bodybuilders": 3200
        }
        calorie_value = calorie_map.get(calorie_intake, 1900)
        
        # Convert supplement checkbox to actual IU value
        supplement_iu = supplement_vitd_iu if takes_supplement else 0
        
        # Prepare user data dictionary
        user_data = {
            'Age': age,
            'Sex': sex,
            'Ethnicity': ethnicity,
            'BMI': weight_kg / ((height_cm / 100) ** 2),
            'supplement_vitd_mcg': supplement_iu / 40,  # Convert IU to mcg
            'dietary_vitd_mcg': dietary_vitd_mcg,
            'calories': calorie_value,
            'smoked_100_cigarettes': smoking_history,
            'vigorous_work': vigorous_activity,
            'moderate_work': moderate_activity,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'takes_glucocorticoid': glucocorticoid,
            'takes_anticonvulsant': anticonvulsant,
            'takes_cholesterol_binder': cholesterol_binder,
            'takes_weightloss_drug': weightloss_drug,
            'takes_vitd_affecting_med': vitd_med
        }
        
        # Engineer features
        engineered = engineer_features(user_data)
        
        # Prepare feature vector
        features = prepare_features(user_data, engineered)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        vit_d_prediction = regression_model.predict(features_scaled)[0]
        vit_d_prediction = max(2, min(168, vit_d_prediction))  # Clamp to realistic range
        
        # Get classification
        classification_probs = classification_model.predict_proba(features_scaled)[0]
        classification = classification_model.predict(features_scaled)[0]
        
        # Get status and interpretation
        status, status_text, color, interpretation = get_vitamin_d_status(vit_d_prediction)
        
        # Create gauge chart
        gauge_fig = create_gauge_chart(vit_d_prediction)
        
        # Create comparison chart
        comparison_fig = create_comparison_chart(vit_d_prediction)
        
        # Build detailed results text
        results_text = f"""
## 📊 Prediction Results

**Predicted Level:** {vit_d_prediction:.1f} ng/mL
**Status:** {status}

**Interpretation:**
{interpretation}

---

### 💡 Recommendations
"""
        
        # Add personalized recommendations
        if vit_d_prediction < 20:
            results_text += """
1. **Immediate Action:** Consult with a healthcare provider about vitamin D supplementation
2. **Sun Exposure:** Aim for 10-30 minutes of midday sun exposure several times per week
3. **Dietary Sources:** Increase intake of fatty fish, fortified milk, and egg yolks
4. **Supplement:** Consider 1000-2000 IU daily (consult doctor for appropriate dose)
"""
        elif vit_d_prediction < 30:
            results_text += """
1. **Increase Vitamin D:** Boost intake through supplements (400-800 IU daily)
2. **Sun Exposure:** Aim for 15-20 minutes of sun exposure 3-4 times per week
3. **Diet:** Include more vitamin D-rich foods in your diet
4. **Monitor:** Recheck levels in 3-6 months
"""
        elif vit_d_prediction < 50:
            results_text += """
1. **Maintain:** Continue your current vitamin D intake
2. **Sun Exposure:** Keep getting regular sun exposure
3. **Diet:** Maintain balanced diet with vitamin D sources
4. **Monitor:** Annual checkups are sufficient
"""
        else:
            results_text += """
1. **Excellent!** Your levels are adequate, maintain current habits
2. **Consistency:** Keep up with regular sun exposure and balanced diet
3. **Monitor:** Annual checkups are recommended
"""
        
        results_text += f"\n\n*Prediction made on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return (
            gr.update(visible=True),  # Show results section
            gauge_fig,
            comparison_fig,
            results_text,
            f"<h3 style='color: {color}; text-align: center; font-size: 28px; margin: 20px 0;'>{status}</h3>"
        )
        
    except Exception as e:
        return (
            gr.update(visible=True),
            None,
            None,
            f"❌ Error: {str(e)}",
            "Error in prediction"
        )

# ==================== GRADIO INTERFACE ====================

# ==================== CUSTOM CSS ====================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --primary-light: #818cf8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark: #1f2937;
    --light: #f9fafb;
}

.gradio-container {
    max-width: 1200px !important;
    background: linear-gradient(135deg, #f3f4f6 0%, #ffffff 100%) !important;
}

/* Header Section */
.header-container {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 60px 40px;
    border-radius: 24px;
    margin-bottom: 40px;
    box-shadow: 0 20px 60px rgba(99, 102, 241, 0.2);
    text-align: center;
}

.header-container h1 {
    font-size: 3em;
    font-weight: 700;
    margin: 0;
    letter-spacing: -1px;
}

.header-container p {
    font-size: 1.2em;
    margin: 15px 0 0 0;
    opacity: 0.95;
    font-weight: 300;
}

/* Form Sections */
.form-section {
    background: white;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
}

.form-section:hover {
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.1);
}

.form-section h2 {
    color: #6366f1;
    font-size: 1.4em;
    margin: 0 0 25px 0;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Input Fields */
input[type="number"], 
input[type="text"],
select,
textarea {
    background: #f9fafb !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    font-size: 1em !important;
    color: #1f2937 !important;
    transition: all 0.3s ease !important;
}

input[type="number"]:focus, 
input[type="text"]:focus,
select:focus,
textarea:focus {
    background: white !important;
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Radio & Checkbox */
input[type="radio"], 
input[type="checkbox"] {
    accent-color: #6366f1 !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
}

/* Radio Group */
.gr-radio {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}

.gr-radio label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: #f3f4f6;
    border-radius: 8px;
    border: 2px solid transparent;
    cursor: pointer;
    transition: all 0.2s ease;
    font-weight: 500;
}

.gr-radio input[type="radio"]:checked + label {
    background: #dbeafe;
    border-color: #6366f1;
    color: #6366f1;
}

/* Button */
button, .submit-button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1.05em !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

button:hover, .submit-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(99, 102, 241, 0.4) !important;
}

button:active, .submit-button:active {
    transform: translateY(0) !important;
}

/* Results Section */
.results-container {
    background: white;
    border-radius: 16px;
    padding: 40px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
}

.results-container h2 {
    color: #1f2937;
    font-size: 1.8em;
    margin: 0 0 30px 0;
    font-weight: 700;
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 16px 32px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1.2em;
    margin: 20px 0;
    text-align: center;
}

.status-deficient {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #991b1b;
    border-left: 5px solid #dc2626;
}

.status-insufficient {
    background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%);
    color: #92400e;
    border-left: 5px solid #f97316;
}

.status-sufficient {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    color: #15803d;
    border-left: 5px solid #10b981;
}

.status-optimal {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    color: #1e40af;
    border-left: 5px solid #3b82f6;
}

/* Chart Container */
.chart-wrapper {
    background: white;
    border-radius: 16px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    border: 1px solid #f3f4f6;
}

/* Info Box */
.info-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border-left: 5px solid #3b82f6;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    color: #1e3a8a;
    font-weight: 500;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #6b7280;
    padding: 30px 20px;
    font-size: 0.95em;
    border-top: 1px solid #e5e7eb;
    margin-top: 40px;
}

.footer-text p {
    margin: 8px 0;
}

/* Responsive */
@media (max-width: 768px) {
    .header-container {
        padding: 40px 20px;
    }
    
    .header-container h1 {
        font-size: 2em;
    }
    
    .form-section {
        padding: 20px;
    }
    
    .results-container {
        padding: 20px;
    }
    
    button, .submit-button {
        width: 100%;
    }
}

/* Markdown adjustments */
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    margin-top: 0 !important;
}

.gr-markdown strong {
    color: #6366f1;
    font-weight: 600;
}

.gr-markdown code {
    background: #f3f4f6;
    color: #6366f1;
    padding: 2px 6px;
    border-radius: 4px;
}
"""

# ==================== GRADIO INTERFACE ====================

def create_interface():
    """Create the Gradio interface with custom styling"""
    
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        # Header
        gr.HTML(
            '<div class="header-container">'
            '<h1>Vitamin D Predictor</h1>'
            '<p>Advanced ML-powered vitamin D level prediction</p>'
            '</div>'
        )
        
        # Info banner
        gr.HTML(
            '<div class="info-box">'
            '<strong>How it works:</strong> Enter your personal, dietary, and lifestyle information below. Our ML model will predict your vitamin D level and provide personalized recommendations.'
            '</div>'
        )
        
        # Form inputs
        with gr.Group(elem_classes="form-section"):
            gr.HTML("<h2>Personal Information</h2>")
            
            with gr.Row():
                age = gr.Number(
                    label="Age (years)",
                    value=30,
                    minimum=1,
                    maximum=120,
                    info="Your age in years"
                )
                sex = gr.Radio(
                    choices=[1, 2],
                    value=1,
                    label="Biological Sex",
                    info="1 = Male, 2 = Female"
                )
            
            ethnicity = gr.Radio(
                choices=[1, 2, 3, 4, 5],
                value=3,
                label="Ethnicity",
                info="1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, 4=Non-Hispanic Black, 5=Other/Mixed"
            )
        
        with gr.Group(elem_classes="form-section"):
            gr.HTML("<h2>Body Measurements</h2>")
            
            with gr.Row():
                height_cm = gr.Number(
                    label="Height (cm)",
                    value=170,
                    minimum=50,
                    maximum=250,
                    info="Your height in centimeters"
                )
                weight_kg = gr.Number(
                    label="Weight (kg)",
                    value=70,
                    minimum=20,
                    maximum=300,
                    info="Your weight in kilograms"
                )
        
        with gr.Group(elem_classes="form-section"):
            gr.HTML("<h2>Dietary Habits</h2>")
            
            takes_supplement = gr.Checkbox(
                label="Do you take vitamin D supplements?",
                value=False,
                info="Check if you use any vitamin D supplements"
            )
            
            supplement_vitd = gr.Slider(
                label="How much vitamin D supplement per day? (IU)",
                value=1000,
                minimum=0,
                maximum=10000,
                step=100,
                info="Typical range: 400-2000 IU",
                visible=False
            )
            
            dietary_freq = gr.Radio(
                choices=[
                    "Rarely (0-1 servings/week)",
                    "Sometimes (2-4 servings/week)",
                    "Often (5-7 servings/week)",
                    "Daily (8+ servings/week)"
                ],
                value="Sometimes (2-4 servings/week)",
                label="How often do you eat vitamin D-rich foods?",
                info="Examples: fatty fish (salmon, mackerel), fortified milk, egg yolks"
            )
            
            calorie_intake = gr.Radio(
                choices=[
                    "Low (1200-1600 kcal) - Small meals, weight loss",
                    "Moderate (1600-2200 kcal) - Average adult",
                    "High (2200-2800 kcal) - Active lifestyle",
                    "Very High (2800+ kcal) - Athletes, bodybuilders"
                ],
                value="Moderate (1600-2200 kcal) - Average adult",
                label="What is your average daily calorie intake?"
            )
            
            # Show/hide supplement input based on checkbox
            takes_supplement.change(
                fn=lambda x: gr.update(visible=x),
                inputs=takes_supplement,
                outputs=supplement_vitd
            )
        
        with gr.Group(elem_classes="form-section"):
            gr.HTML("<h2>Lifestyle & Activity</h2>")
            
            with gr.Row():
                smoking = gr.Radio(
                    choices=[1, 2],
                    value=2,
                    label="Smoking History",
                    info="1=Smoked 100+ cigarettes, 2=Never or less"
                )
                vigorous = gr.Radio(
                    choices=[1, 2],
                    value=2,
                    label="Vigorous Work Activity",
                    info="1=Yes, 2=No"
                )
                moderate = gr.Radio(
                    choices=[1, 2],
                    value=2,
                    label="Moderate Work Activity",
                    info="1=Yes, 2=No"
                )
        
        with gr.Group(elem_classes="form-section"):
            gr.HTML("<h2>Health Information (Optional)</h2>")
            gr.HTML("<p style='color: #6b7280; font-size: 0.95em;'>These fields help improve prediction accuracy. Default values are provided.</p>")
            
            with gr.Row():
                systolic = gr.Number(
                    label="Systolic Blood Pressure (mmHg)",
                    value=120,
                    minimum=70,
                    maximum=200,
                    info="Upper blood pressure reading"
                )
                diastolic = gr.Number(
                    label="Diastolic Blood Pressure (mmHg)",
                    value=80,
                    minimum=40,
                    maximum=120,
                    info="Lower blood pressure reading"
                )
            
            gr.HTML("<p style='font-weight: 600; margin-bottom: 10px;'>Medications (check if applicable):</p>")
            with gr.Row():
                med_glucocorticoid = gr.Checkbox(
                    label="Glucocorticoids (e.g., prednisone)",
                    value=False
                )
                med_anticonvulsant = gr.Checkbox(
                    label="Anticonvulsants (e.g., phenytoin)",
                    value=False
                )
                med_cholesterol = gr.Checkbox(
                    label="Cholesterol Binders (e.g., cholestyramine)",
                    value=False
                )
            
            with gr.Row():
                med_weightloss = gr.Checkbox(
                    label="Weight Loss Drugs (e.g., orlistat)",
                    value=False
                )
                med_vitd = gr.Checkbox(
                    label="Other Vitamin D-Affecting Medication",
                    value=False
                )
        
        # Predict button
        predict_btn = gr.Button(
            "Predict My Vitamin D Level",
            scale=2,
            size="lg"
        )
        
        # Results section (initially hidden)
        results_visible = gr.State(False)
        
        with gr.Group(elem_classes="results-container", visible=False) as results_group:
            gr.HTML("<h2>Your Results</h2>")
            
            status_display = gr.HTML("")
            
            with gr.Row():
                gauge_chart = gr.Plot(label="Vitamin D Level Gauge")
            
            with gr.Row():
                comparison_chart = gr.Plot(label="Level Comparison")
            
            results_text = gr.Markdown()
        
        # Connect button to prediction function
        predict_btn.click(
            predict_vitamin_d,
            inputs=[
                age, sex, ethnicity, height_cm, weight_kg,
                takes_supplement, supplement_vitd, dietary_freq, calorie_intake,
                smoking, vigorous, moderate,
                systolic, diastolic,
                med_glucocorticoid, med_anticonvulsant, med_cholesterol,
                med_weightloss, med_vitd
            ],
            outputs=[
                results_group, gauge_chart, comparison_chart, results_text, status_display
            ]
        )
        
        # Footer
        gr.HTML(
            '<div class="footer-text">'
            '<p><strong>Disclaimer:</strong> This tool provides estimates based on machine learning models. It is not a substitute for professional medical advice.</p>'
            '<p style="font-size: 0.9em; color: #9ca3af;">Dataset: NHANES 2005-2018 | Model: LightGBM + XGBoost | Last Updated: January 2026</p>'
            '<p style="font-size: 0.85em; color: #6b7280;"><a href="https://github.com/ScriptSorcerer23" target="_blank" style="color: #3b82f6; text-decoration: none;">GitHub: ScriptSorcerer23</a></p>'
            '</div>'
        )
    
    return demo

# ==================== MAIN ====================

if __name__ == "__main__":
    print("🌿 Starting Vitamin D Prediction Interface...")
    print("Loading models...")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
