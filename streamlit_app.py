
import streamlit as st

from app.predict import predict_loan


st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Approval & Amount Predictor")
st.caption(
    "Two-stage machine learning pipeline for "
    "loan approval and amount prediction"
)

st.info(
    "Educational project using synthetic data. "
    "Predictions are not suitable for actual lending decisions."
)

st.divider()

st.subheader("Applicant Information")

with st.form("loan_application"):
    st.markdown("#### Personal details")

    col1, col2, col3 = st.columns(3)

    with col1:
        dependents = st.number_input(
            "Number of dependents",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )

    with col2:
        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"],
        )

    with col3:
        self_employed = st.selectbox(
            "Self-employed",
            ["No", "Yes"],
        )

    st.markdown("#### Financial details")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input(
            "Annual income (₹)",
            min_value=0,
            value=1200000,
            step=10000,
        )

    with col2:
        requested_amount = st.number_input(
            "Requested loan amount (₹)",
            min_value=1,
            value=30000,
            step=1000,
        )

    with col3:
        loan_term = st.number_input(
            "Loan term (years)",
            min_value=1,
            max_value=50,
            value=12,
            step=1,
        )

    st.markdown("#### Credit and assets")

    col1, col2, col3 = st.columns(3)

    with col1:
        cibil_score = st.number_input(
            "CIBIL score",
            min_value=300,
            max_value=900,
            value=800,
            step=1,
        )

    with col2:
        residential_assets = st.number_input(
            "Residential assets (₹)",
            min_value=0,
            value=2000000,
            step=100000,
        )

    with col3:
        commercial_assets = st.number_input(
            "Commercial assets (₹)",
            min_value=0,
            value=2000000,
            step=100000,
        )

    col1, col2 = st.columns(2)

    with col1:
        luxury_assets = st.number_input(
            "Luxury assets (₹)",
            min_value=0,
            value=0,
            step=100000,
        )

    with col2:
        bank_assets = st.number_input(
            "Bank assets (₹)",
            min_value=0,
            value=55000,
            step=10000,
        )

    submitted = st.form_submit_button(
        "Predict Loan Approval",
        type="primary",
        use_container_width=True,
    )


if submitted:
    application = {
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income,
        "requested_loan_amount": requested_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets,
        "commercial_assets_value": commercial_assets,
        "luxury_assets_value": luxury_assets,
        "bank_asset_value": bank_assets,
    }

    try:
        with st.spinner("Analyzing application..."):
            result = predict_loan(application)

        st.divider()
        st.subheader("Prediction Results")

        if result["loan_status"] == "Approved":
            st.success("Predicted status: Approved")

            amount = result["predicted_loan_amount"]

            st.metric(
                "Predicted loan amount",
                f"₹{amount:,.2f}",
            )

            if amount > requested_amount:
                st.warning(
                    "The predicted amount exceeds the "
                    "requested amount. Review this model output."
                )
        else:
            st.error("Predicted status: Rejected")
            st.write(
                "No loan amount prediction is provided "
                "because the application was classified "
                "as rejected."
            )

        st.caption(
            "These are model predictions based on synthetic "
            "training data, not actual approval decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")