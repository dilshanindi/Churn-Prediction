import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("xgboost_model (1).pkl")

# Set page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title (centered at the top)
st.markdown("<h1 style='text-align: center; color: white;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Subtitle / Description
st.markdown("<p style='text-align: center; color: gray;'>Enter customer details to predict churn.</p>", unsafe_allow_html=True)

# # Centered image (smaller size)
# image_path = "churn banner.jpg"  # Ensure the image file is in the same directory
# st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
# st.image(image_path, width=400)  # Adjust width to make it smaller
# st.markdown("</div>", unsafe_allow_html=True)


# Sidebar for user input
st.sidebar.header("Enter Customer Data")

def get_user_input():
    contract_options = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    tech_support_options = {"No": 0, "Yes": 1}
    dependents_options = {"No": 0, "Yes": 1}

    contract = st.sidebar.selectbox("Contract Type", list(contract_options.keys()))
    tech_support = st.sidebar.selectbox("Tech Support", list(tech_support_options.keys()))
    dependents = st.sidebar.selectbox("Dependents", list(dependents_options.keys()))
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)

    # Create DataFrame
    data = pd.DataFrame({
        "Contract": [contract_options[contract]],
        "Tech Support": [tech_support_options[tech_support]],
        "Dependents": [dependents_options[dependents]],
        "Monthly Charges": [monthly_charges]
    })

    return data

# Get user input
input_data = get_user_input()

# Placeholder for result (this helps with auto-scrolling)
st.markdown('<div id="result"></div>', unsafe_allow_html=True)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Churn probability

    # Auto-scroll to prediction result using JavaScript
    st.markdown(
        """
        <script>
        var result = document.getElementById("result");
        result.scrollIntoView({behavior: "smooth"});
        </script>
        """,
        unsafe_allow_html=True
    )

    # Display prediction
    st.subheader("Prediction Result")

    # Highlight Prediction Outcome
    if prediction == 1:
        st.markdown("<h2 style='text-align: center; color: red; font-size: 28px;'>🔴 Churn</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: green; font-size: 28px;'>🟢 No Churn</h2>", unsafe_allow_html=True)

    # Show churn probability in bold
    st.markdown(f"""
        <div style="text-align: center; font-size: 20px;">
            <b>Churn Probability: {probability:.2%}</b>
        </div>
    """, unsafe_allow_html=True)
