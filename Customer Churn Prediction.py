import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import google.generativeai as genai
import base64
from typing import Dict, List, Tuple

# Configure page
st.set_page_config(
    page_title="Customer Loyalty Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .churn-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stay-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'batch_recommendations' not in st.session_state:
    st.session_state.batch_recommendations = None
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'current_customer_data' not in st.session_state:
    st.session_state.current_customer_data = None
if 'current_explanation' not in st.session_state:
    st.session_state.current_explanation = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'batch_predictions_done' not in st.session_state:
    st.session_state.batch_predictions_done = False



@st.cache_resource
def load_model():
    """Load the pre-trained churn prediction model."""
    try:
        with open('churn_prediction_pipeline_gridsearch_tuned_final.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(
            "Model file 'churn_prediction_pipeline_gridsearch_tuned_final.pkl' not found. Please ensure the model file is in the same directory as this application.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def get_feature_names() -> List[str]:
    """Return the exact feature names in the correct order."""
    return [
        'Tenure in Months',
        'Monthly Charge',
        'Number of Referrals',
        'Contract_Two Year',
        'Internet Type_Fiber Optic',
        'Total Long Distance Charges',
        'Age',
        'Number of Dependents',
        'Paperless Billing_Yes',
        'Payment Method_Credit Card',
        'Contract_One Year',
        'Total Revenue'
    ]


def create_input_dataframe(user_inputs: Dict) -> pd.DataFrame:
    """Create a DataFrame from user inputs with correct feature names and order."""
    feature_names = get_feature_names()

    # Create DataFrame with features in correct order
    data = []
    for feature in feature_names:
        data.append(user_inputs[feature])

    df = pd.DataFrame([data], columns=feature_names)
    return df


def predict_churn(model, customer_data: pd.DataFrame) -> Tuple[float, str]:
    """Make churn prediction for a single customer."""
    try:
        # Get probability prediction
        probability = model.predict_proba(customer_data)[0][1]  # Probability of churn (class 1)

        # Apply threshold
        prediction = "CHURN" if probability >= 0.5 else "STAY"

        return probability, prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return 0.0, "ERROR"


def get_feature_importance_explanation(model, customer_data: pd.DataFrame, feature_names: List[str]) -> str:
    """Get top factors influencing the prediction using feature importance."""
    try:
        # Get feature importance from XGBoost model
        # Note: This is a simplified approach since we don't have SHAP values
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # If using pipeline, get the classifier step
            classifier = model.named_steps.get('classifier') or model.steps[-1][1]
            importances = classifier.feature_importances_

        # Get top 3 most important features
        top_indices = np.argsort(importances)[-3:][::-1]

        explanations = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            feature_value = customer_data.iloc[0, idx]
            importance = importances[idx]

            # Create human-readable explanations
            if feature_name == 'Tenure in Months':
                if feature_value <= 12:
                    explanations.append(f"Short tenure ({feature_value} months)")
                elif feature_value >= 36:
                    explanations.append(f"Long tenure ({feature_value} months)")
                else:
                    explanations.append(f"Medium tenure ({feature_value} months)")

            elif feature_name == 'Monthly Charge':
                if feature_value >= 80:
                    explanations.append(f"High monthly charge (${feature_value:.2f})")
                elif feature_value <= 40:
                    explanations.append(f"Low monthly charge (${feature_value:.2f})")
                else:
                    explanations.append(f"Medium monthly charge (${feature_value:.2f})")

            elif feature_name == 'Contract_Two Year':
                explanations.append("Has Two Year Contract" if feature_value == 1 else "No Two Year Contract")

            elif feature_name == 'Contract_One Year':
                explanations.append("Has One Year Contract" if feature_value == 1 else "No One Year Contract")

            elif feature_name == 'Number of Referrals':
                if feature_value == 0:
                    explanations.append("No referrals made")
                elif feature_value >= 5:
                    explanations.append(f"High referrals ({feature_value})")
                else:
                    explanations.append(f"Some referrals ({feature_value})")

            else:
                explanations.append(f"{feature_name}: {feature_value}")

        return f"This customer's prediction is primarily driven by: {', '.join(explanations[:3])}"

    except Exception as e:
        return f"Unable to generate feature explanation: {str(e)}"


def generate_batch_recommendations(churn_data: pd.DataFrame, gemini_api_key: str) -> str:
    """Generate AI-powered batch recommendations for high-risk customers."""
    try:
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Updated model name

        # Validate input data
        if churn_data.empty:
            return "Error: No data provided for analysis."

        # Check if required columns exist
        required_columns = ['Predicted Churn', 'Predicted Churn Probability']
        missing_columns = [col for col in required_columns if col not in churn_data.columns]
        if missing_columns:
            return f"Error: Missing required columns: {missing_columns}"

        # Analyze the batch data
        total_customers = len(churn_data)

        # Handle different churn prediction formats (Yes/No vs 1/0 vs True/False)
        if churn_data['Predicted Churn'].dtype == 'object':
            high_risk_customers = churn_data[churn_data['Predicted Churn'].isin(['Yes', 'CHURN', '1', 1, True])]
        else:
            high_risk_customers = churn_data[churn_data['Predicted Churn'] == 1]

        if len(high_risk_customers) == 0:
            return "Great news! No high-risk customers identified in this batch. Focus on maintaining current satisfaction levels and implementing proactive retention strategies."

        churn_rate = len(high_risk_customers) / total_customers * 100

        # Calculate key statistics with error handling
        avg_churn_prob = high_risk_customers['Predicted Churn Probability'].mean()

        # Check if optional columns exist before calculating
        stats_info = []

        if 'Monthly Charge' in high_risk_customers.columns:
            high_monthly_charge = high_risk_customers['Monthly Charge'].mean()
            stats_info.append(f"Average Monthly Charge (High-Risk): ${high_monthly_charge:.2f}")

        if 'Tenure in Months' in high_risk_customers.columns:
            low_tenure = high_risk_customers['Tenure in Months'].mean()
            stats_info.append(f"Average Tenure (High-Risk): {low_tenure:.1f} months")

        if 'Internet Type_Fiber Optic' in high_risk_customers.columns:
            fiber_optic_churners = high_risk_customers['Internet Type_Fiber Optic'].sum()
            stats_info.append(f"Fiber Optic Customers at Risk: {fiber_optic_churners}")

        if 'Contract_Two Year' in high_risk_customers.columns and 'Contract_One Year' in high_risk_customers.columns:
            no_contract_churners = len(high_risk_customers[
                                           (high_risk_customers['Contract_Two Year'] == 0) &
                                           (high_risk_customers['Contract_One Year'] == 0)
                                           ])
            stats_info.append(f"Month-to-Month Contract Customers at Risk: {no_contract_churners}")

        # Create the prompt
        stats_text = "\n- ".join(stats_info) if stats_info else "Additional customer details not available in dataset"

        prompt = f"""
        You are a senior data analyst specializing in telecom customer retention strategies.

        BATCH ANALYSIS SUMMARY:
        - Total Customers Analyzed: {total_customers}
        - High-Risk Customers (Churn Predicted): {len(high_risk_customers)}
        - Overall Churn Rate: {churn_rate:.1f}%
        - Average Churn Probability: {avg_churn_prob:.1%}

        KEY RISK PATTERNS IDENTIFIED:
        - {stats_text}

        Please provide concise, actionable recommendations in the following format:

        **STRATEGIC RECOMMENDATIONS** (3-4 company-wide initiatives)
        1. [Recommendation 1]
        2. [Recommendation 2]
        3. [Recommendation 3]

        **TACTICAL ACTIONS** (3-4 immediate steps for high-risk customers)
        1. [Action 1]
        2. [Action 2]
        3. [Action 3]

        **PREVENTIVE MEASURES** (2-3 proactive strategies)
        1. [Measure 1]
        2. [Measure 2]

        **SUCCESS METRICS** (2-3 KPIs to track)
        1. [Metric 1]
        2. [Metric 2]

        Keep each point concise (1-2 sentences) and focus on actionable business recommendations.
        """

        # Generate content with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=1000,
                        temperature=0.7,
                    )
                )

                # Check if response was generated successfully
                if response and response.text:
                    return response.text
                else:
                    if attempt == max_retries - 1:
                        return "Error: Gemini API returned empty response after multiple attempts."
                    continue

            except Exception as api_error:
                if attempt == max_retries - 1:
                    return f"Error calling Gemini API: {str(api_error)}. Please check your API key and quota."
                continue

    except Exception as e:
        return f"Error generating batch recommendations: {str(e)}. Please check your data format and API configuration."

def generate_recommendations(churn_probability: float, explanation: str, gemini_api_key: str) -> str:
    """Generate AI-powered recommendations using Gemini API."""
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = f"""
        You are an AI assistant specialized in telecom customer retention. 

        Customer Profile:
        - Churn Probability: {churn_probability:.1%}
        - Key Risk Factors: {explanation}

        Based on this customer profile and churn drivers, suggest 3-5 actionable strategies to retain this customer. 
        Be concise, practical, and focus on specific actions that can be implemented immediately.
        Format your response as numbered points.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Unable to generate recommendations: {str(e)}. Please check your API key and try again."


def process_batch_predictions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Process batch predictions for uploaded CSV."""
    try:
        # Make predictions
        probabilities = model.predict_proba(df)[:, 1]
        predictions = ["Yes" if p >= 0.5 else "No" for p in probabilities]

        # Add results to dataframe
        result_df = df.copy()
        result_df['Predicted Churn Probability'] = probabilities
        result_df['Predicted Churn'] = predictions

        return result_df

    except Exception as e:
        st.error(f"Error processing batch predictions: {str(e)}")
        return pd.DataFrame()


def download_csv(df: pd.DataFrame, filename: str) -> str:
    """Create download link for CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href


def main():
    # Header
    st.markdown('<div class="main-header">🧭 Customer Loyalty Compass</div>', unsafe_allow_html=True)
    st.markdown("**Churn Prediction & Actionable Insights**")

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        gemini_api_key = st.text_input("Gemini API Key", type="password",
                                       help="Enter your Gemini API key for AI recommendations")
        st.markdown("---")
        st.markdown("**About this App**")
        st.markdown(
            "This application predicts customer churn using machine learning and provides actionable retention strategies.")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Predict Single Customer", "📊 Batch Prediction", "🌐 Global Insights"])

    with tab1:
        st.header("Predict Churn for a Single Customer")

        # Create input form
        with st.form("single_prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Customer Demographics")
                age = st.number_input("Age", min_value=18, max_value=100, value=90)
                tenure = st.number_input("Tenure in Months", min_value=0, max_value=100, value=1)
                dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
                referrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=1)

                st.subheader("Contract & Billing")
                contract_two_year = st.checkbox("Two Year Contract")
                contract_one_year = st.checkbox("One Year Contract")
                paperless_billing = st.checkbox("Paperless Billing")
                credit_card_payment = st.checkbox("Credit Card Payment")

            with col2:
                st.subheader("Service & Charges")
                monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=200.0, value=35.0,
                                                 step=0.01)
                total_revenue = st.number_input("Total Revenue ($)", min_value=0.0, max_value=10000.0, value=80.0,
                                                step=0.01)
                long_distance_charges = st.number_input("Total Long Distance Charges ($)", min_value=0.0,
                                                        max_value=1000.0, value=50.0, step=0.01)

                st.subheader("Internet Service")
                fiber_optic = st.checkbox("Fiber Optic Internet")

            # Submit button
            submitted = st.form_submit_button("Predict Churn", type="primary")

        if submitted:
            # Prepare input data
            user_inputs = {
                'Tenure in Months': tenure,
                'Monthly Charge': monthly_charge,
                'Number of Referrals': referrals,
                'Contract_Two Year': int(contract_two_year),
                'Internet Type_Fiber Optic': int(fiber_optic),
                'Total Long Distance Charges': long_distance_charges,
                'Age': age,
                'Number of Dependents': dependents,
                'Paperless Billing_Yes': int(paperless_billing),
                'Payment Method_Credit Card': int(credit_card_payment),
                'Contract_One Year': int(contract_one_year),
                'Total Revenue': total_revenue
            }

            # Create DataFrame
            customer_df = create_input_dataframe(user_inputs)

            # Make prediction
            probability, prediction = predict_churn(model, customer_df)

            # Store in session state
            st.session_state.last_prediction = (probability, prediction)
            st.session_state.current_customer_data = customer_df
            st.session_state.current_explanation = get_feature_importance_explanation(model, customer_df,
                                                                                      get_feature_names())

        # Display results if prediction exists
        if st.session_state.last_prediction is not None:
            probability, prediction = st.session_state.last_prediction
            customer_df = st.session_state.current_customer_data
            explanation = st.session_state.current_explanation

            # Display results
            st.subheader("Prediction Results")

            if prediction == "CHURN":
                st.markdown(f"""
                <div class="churn-alert">
                    <h3 style="color: #f44336; margin: 0;">⚠️ Customer is likely to CHURN</h3>
                    <p style="margin: 0.5rem 0 0 0;">Predicted Churn Probability: <strong>{probability:.1%}</strong></p>
                    <small>Prediction based on >50% probability of churn</small>
                </div>
                """, unsafe_allow_html=True)

                # Feature importance explanation
                st.subheader("Key Factors for This Prediction")
                st.write(explanation)

                # AI Recommendations
                st.subheader("🎯 Retention Action Plan")

                # Create columns for the recommendation section
                rec_col1, rec_col2 = st.columns([3, 1])

                with rec_col1:
                    # Generate recommendations button (outside of form)
                    if st.button("🧠 Generate AI Recommendations", type="secondary", key="gen_recommendations"):
                        if not gemini_api_key:
                            st.warning(
                                "⚠️ Please enter your Gemini API key in the sidebar to generate recommendations.")
                        else:
                            with st.spinner("🤖 Generating personalized recommendations..."):
                                recommendations = generate_recommendations(probability, explanation, gemini_api_key)
                                st.session_state.recommendations = recommendations

                with rec_col2:
                    # Add a button to clear recommendations
                    if st.session_state.recommendations:
                        if st.button("🗑️ Clear", key="clear_recommendations"):
                            st.session_state.recommendations = None
                            st.rerun()

                # Display recommendations if they exist
                if st.session_state.recommendations:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 15px;
                        margin: 20px 0;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h3 style="color: white; margin-top: 0; display: flex; align-items: center;">
                            🎯 AI-Powered Retention Strategies
                        </h3>
                        <div style='background-color: #ffffff; padding: 20px; border-radius: 10px;
                                        box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-top: 10px;'>
                                {st.session_state.recommendations.replace('\n', '<br>')}
                            </div>
                    """, unsafe_allow_html=True)

                    st.markdown("</div></div>", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="stay-alert">
                    <h3 style="color: #4caf50; margin: 0;">✅ Customer is likely to STAY</h3>
                    <p style="margin: 0.5rem 0 0 0;">Predicted Churn Probability: <strong>{probability:.1%}</strong></p>
                    <small>Prediction based on <50% probability of churn</small>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.header("Batch Churn Prediction (CSV Upload)")

        st.info("""
        **Important:** Please ensure your CSV file contains the following 12 columns in the correct numerical/binary format:

        1. Tenure in Months (integer)
        2. Monthly Charge (float)
        3. Number of Referrals (integer)
        4. Contract_Two Year (0 or 1)
        5. Internet Type_Fiber Optic (0 or 1)
        6. Total Long Distance Charges (float)
        7. Age (integer)
        8. Number of Dependents (integer)
        9. Paperless Billing_Yes (0 or 1)
        10. Payment Method_Credit Card (0 or 1)
        11. Contract_One Year (0 or 1)
        12. Total Revenue (float)

        Missing values should be handled prior to upload.
        """)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)

                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Validate columns
                expected_columns = get_feature_names()
                if not all(col in df.columns for col in expected_columns):
                    st.error(
                        "CSV file does not contain all required columns. Please check the column names and format.")
                else:
                    if st.button("Predict Churn for Uploaded File", type="primary"):
                        with st.spinner("Processing predictions..."):
                            # Process predictions
                            results_df = process_batch_predictions(model, df[expected_columns])

                            if not results_df.empty:
                                # Summary statistics
                                st.session_state.results_df = results_df
                                st.session_state.batch_predictions_done = True

                    # ✅ This block shows the results persistently
                    if st.session_state.batch_predictions_done and st.session_state.results_df is not None:
                        results_df = st.session_state.results_df

                        st.subheader("Prediction Summary")
                        total_customers = len(results_df)
                        churn_customers = len(results_df[results_df['Predicted Churn'] == 'Yes'])
                        churn_rate = (churn_customers / total_customers) * 100

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", total_customers)
                        with col2:
                            st.metric("Predicted Churners", churn_customers)
                        with col3:
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")

                        st.subheader("Detailed Results")
                        st.dataframe(results_df, use_container_width=True)

                        st.markdown("### 📥 Download Results")
                        csv_download = download_csv(results_df, "churn_predictions.csv")
                        st.markdown(csv_download, unsafe_allow_html=True)

                        # Segmentation analysis
                        st.markdown("### 📈 Risk Segmentation Analysis")

                        high_risk = results_df[results_df['Predicted Churn Probability'] >= 0.7]
                        medium_risk = results_df[(results_df['Predicted Churn Probability'] >= 0.5) &
                                                 (results_df['Predicted Churn Probability'] < 0.7)]
                        low_risk = results_df[results_df['Predicted Churn Probability'] < 0.5]

                        seg_col1, seg_col2, seg_col3 = st.columns(3)

                        with seg_col1:
                            st.markdown(f"""
                            <div style="background-color: #ffebee; border: 2px solid #f44336; border-radius: 10px; padding: 15px; text-align: center;">
                                <h4 style="color: #f44336; margin: 0;">🚨 HIGH RISK</h4>
                                <h2 style="color: #f44336; margin: 5px 0;">{len(high_risk)}</h2>
                                <p style="margin: 0; color: #666;">Churn Prob ≥ 70%</p>
                                <p style="margin: 5px 0; color: #666;"><strong>Action:</strong> Immediate intervention</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with seg_col2:
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; border: 2px solid #ff9800; border-radius: 10px; padding: 15px; text-align: center;">
                                <h4 style="color: #ff9800; margin: 0;">⚠️ MEDIUM RISK</h4>
                                <h2 style="color: #ff9800; margin: 5px 0;">{len(medium_risk)}</h2>
                                <p style="margin: 0; color: #666;">Churn Prob 50-70%</p>
                                <p style="margin: 5px 0; color: #666;"><strong>Action:</strong> Proactive retention</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with seg_col3:
                            st.markdown(f"""
                            <div style="background-color: #e8f5e8; border: 2px solid #4caf50; border-radius: 10px; padding: 15px; text-align: center;">
                                <h4 style="color: #4caf50; margin: 0;">✅ LOW RISK</h4>
                                <h2 style="color: #4caf50; margin: 5px 0;">{len(low_risk)}</h2>
                                <p style="margin: 0; color: #666;">Churn Prob < 50%</p>
                                <p style="margin: 5px 0; color: #666;"><strong>Action:</strong> Maintain satisfaction</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Batch Retention Strategy section
                        st.markdown("### 🎯 Batch Retention Strategy")

                        batch_rec_col1, batch_rec_col2 = st.columns([3, 1])
                        with batch_rec_col1:
                            if st.button("🧠 Generate Batch Recommendations", type="secondary", key="gen_batch_rec"):
                                if not gemini_api_key:
                                    st.warning("⚠️ Please enter your Gemini API key in the sidebar.")
                                else:
                                    with st.spinner("🤖 Generating recommendations..."):
                                        batch_recommendations = generate_batch_recommendations(
                                            results_df,
                                            gemini_api_key
                                        )
                                        st.session_state.batch_recommendations = batch_recommendations

                        with batch_rec_col2:
                            if st.session_state.batch_recommendations:
                                if st.button("🗑️ Clear", key="clear_batch_rec"):
                                    st.session_state.batch_recommendations = None
                                    st.rerun()

                        if st.session_state.batch_recommendations:
                            st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 20px;
                                    border-radius: 15px;
                                    margin: 20px 0;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                ">
                                    <h3 style="color: white; margin-top: 0; display: flex; align-items: center;">
                                        📊 Strategic Retention Plan for Batch
                                    </h3>
                                    <div style='background-color: #ffffff; padding: 20px; border-radius: 10px;
                                                box-shadow: 0 0 10px rgba(0,0,0,0.1); margin-top: 10px;'>
                                        {st.session_state.batch_recommendations}
                            """, unsafe_allow_html=True)



            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    with tab3:
        st.header("📊 Global Insights – Key Factors Influencing Customer Churn")

        st.subheader("What We Discovered")
        st.markdown("""
        After analyzing customer behavior and patterns from historical data, these are the **top factors** that influence whether a customer stays or leaves:

        **1. Contract Type**
        - Customers with long-term contracts (1 or 2 years) are significantly more likely to stay.
        - Month-to-month customers show a much higher risk of leaving.

        **2. Payment Method**
        - Customers who pay using **credit cards** are generally more loyal, possibly due to convenience or autopay setups.

        **3. Referrals & Family Commitments**
        - Customers who have **referred others** or have **dependents** tend to be more engaged and less likely to churn.

        **4. Tenure & Engagement**
        - Newer customers or those with **low tenure** are more likely to leave, especially in the first 6–12 months.
        - Customers with **higher monthly charges** but low perceived value may also churn if not supported well.

        **5. Internet Type**
        - A large number of high-risk customers are using **fiber optic internet**. This could signal dissatisfaction or competition in that segment.
        """)

        st.subheader("🧠 Feature Importance Chart")

        features = [
            'Contract_Two Year', 'Contract_One Year', 'Payment Method_Credit Card',
            'Number of Referrals', 'Number of Dependents', 'Tenure in Months',
            'Monthly Charge', 'Age', 'Internet Type_Fiber Optic',
            'Total Revenue', 'Total Long Distance Charges', 'Paperless Billing_Yes'
        ]
        importance_scores = [
            0.402612, 0.298387, 0.136708,
            0.056108, 0.030286, 0.017538,
            0.014012, 0.011670, 0.010829,
            0.010146, 0.006558, 0.005147
        ]

        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(features[::-1], importance_scores[::-1], color='#4e79a7')
        ax.set_xlabel('Importance Score')
        ax.set_title('Top 12 Factors Contributing to Customer Churn', fontsize=14)
        ax.set_xlim(0, max(importance_scores) + 0.05)

        for bar, score in zip(bars, importance_scores[::-1]):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{score:.3f}', ha='left', va='center', fontsize=9)

        st.pyplot(fig)

        st.markdown("""
        ### 🎯 Strategic Focus Areas
        - **Retention starts at onboarding** — invest in early engagement during the first few months.
        - **Reward loyalty** — offer perks or discounts for contract renewals and referrals.
        - **Monitor high spenders** — ensure that customers with high bills feel they’re getting value.
        - **Understand household context** — families may have stronger retention potential if supported properly.
        """)


if __name__ == "__main__":
    # Check if running directly or through streamlit
    try:
        # Try to import streamlit context
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            # Running directly, launch streamlit
            import subprocess
            import sys

            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        else:
            # Running through streamlit
            main()
    except ImportError:
        # Fallback to direct launch
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    except Exception:
        # If all else fails, just run main
        main()
