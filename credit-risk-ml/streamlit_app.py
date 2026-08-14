import streamlit as st
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Credit Decisioning", page_icon="🏦", layout="wide")

session = get_active_session()

st.title("Credit Default Risk Assessment")
st.markdown("Predict probability of loan default using bureau features. Model: XGBoost trained on Experian Premier Attributes, ClearView, and vendor scores.")

tab1, tab2 = st.tabs(["Single Applicant", "Batch Scoring"])

with tab1:
    st.subheader("Applicant Bureau Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Vendor Scores**")
        fico_score = st.slider("FICO Score", 300, 850, 680)
        vantage_score = st.slider("Vantage Score", 300, 850, 670)
        vendor_bankruptcy_score = st.slider("Vendor Bankruptcy Score", 1, 999, 500)
    
    with col2:
        st.markdown("**Premier Attributes**")
        num_open_trades = st.number_input("Open Trades", 1, 30, 8)
        num_trades_ever = st.number_input("Total Trades Ever", 2, 60, 15)
        revolving_util = st.slider("Revolving Utilization %", 0.0, 100.0, 45.0)
        revolving_balance = st.number_input("Revolving Balance ($)", 0, 200000, 12000)
        installment_balance = st.number_input("Installment Balance ($)", 0, 300000, 25000)
        months_oldest = st.number_input("Months Since Oldest Trade", 6, 480, 180)
    
    with col3:
        st.markdown("**Delinquency & Inquiries**")
        delinq_30 = st.number_input("30-Day Delinquencies (12M)", 0, 8, 0)
        delinq_60 = st.number_input("60-Day Delinquencies (12M)", 0, 5, 0)
        delinq_90 = st.number_input("90+ Day Delinquencies (Ever)", 0, 10, 0)
        inquiries_6m = st.number_input("Inquiries (6M)", 0, 12, 1)
        months_newest = st.number_input("Months Since Newest Trade", 0, 60, 12)
        bankruptcy_flag = st.selectbox("Bankruptcy on Record", [0, 1], index=0)
        
    st.markdown("**ClearView Trended Data**")
    cv1, cv2 = st.columns(2)
    with cv1:
        payment_velocity = st.slider("Payment Velocity Trend", -1.0, 1.0, 0.1, help="-1 = declining payments, +1 = improving")
    with cv2:
        balance_trend = st.slider("Balance Trend (12M)", -1.0, 1.0, 0.0, help="-1 = balances falling (good), +1 = rising")
    
    if st.button("Score Applicant", type="primary"):
        query = f"""
        SELECT CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
            {fico_score}, {vantage_score}, {num_open_trades}, {num_trades_ever},
            {revolving_util}, {revolving_balance},
            {delinq_30}, {delinq_60}, {delinq_90},
            {months_oldest}, {months_newest}, {inquiries_6m},
            {bankruptcy_flag}, {installment_balance},
            {payment_velocity}, {balance_trend}, {vendor_bankruptcy_score}
        ) AS PREDICTION
        """
        result = session.sql(query).collect()
        prediction = result[0]["PREDICTION"]
        
        import json
        pred_dict = json.loads(prediction)
        prob_default = pred_dict.get("output_feature_1", 0)
        
        st.divider()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Probability of Default", f"{prob_default:.1%}")
        with c2:
            decision = "DECLINE" if prob_default > 0.35 else "APPROVE"
            color = "red" if decision == "DECLINE" else "green"
            st.markdown(f"### Decision: :{color}[{decision}]")
        with c3:
            risk_tier = (
                "Very High" if prob_default > 0.7 else
                "High" if prob_default > 0.5 else
                "Medium" if prob_default > 0.35 else
                "Low" if prob_default > 0.15 else
                "Very Low"
            )
            st.metric("Risk Tier", risk_tier)
        
        if prob_default > 0.35:
            st.warning(f"Applicant exceeds the 35% default probability threshold. Recommended action: DECLINE or refer to manual review.")
        else:
            st.success(f"Applicant within acceptable risk parameters. Recommended action: APPROVE.")

with tab2:
    st.subheader("Batch Scoring Results")
    st.markdown("Score all applicants in `CREDIT_RISK.ML.LOAN_APPLICATIONS` and view decisions.")
    
    n_rows = st.slider("Number of applicants to score", 10, 100, 20)
    
    if st.button("Run Batch Scoring"):
        batch_query = f"""
        WITH scored AS (
            SELECT 
                APPLICANT_ID,
                FICO_SCORE,
                REVOLVING_UTILIZATION_PCT,
                NUM_DELINQ_30_12M,
                CREDIT_RISK.ML.CREDIT_DEFAULT_MODEL!PREDICT_PROBA(
                    FICO_SCORE, VANTAGE_SCORE, NUM_OPEN_TRADES, NUM_TRADES_EVER,
                    REVOLVING_UTILIZATION_PCT, TOTAL_REVOLVING_BALANCE,
                    NUM_DELINQ_30_12M, NUM_DELINQ_60_12M, NUM_DELINQ_90_EVER,
                    MONTHS_SINCE_OLDEST_TRADE, MONTHS_SINCE_NEWEST_TRADE, NUM_INQUIRIES_6M,
                    BANKRUPTCY_FLAG, TOTAL_INSTALLMENT_BALANCE,
                    PAYMENT_VELOCITY_TREND, BALANCE_TREND_12M, VENDOR_BANKRUPTCY_SCORE
                ) AS PREDICTION,
                DEFAULT_18M AS ACTUAL_DEFAULT
            FROM CREDIT_RISK.ML.LOAN_APPLICATIONS
            LIMIT {n_rows}
        )
        SELECT 
            APPLICANT_ID,
            FICO_SCORE,
            REVOLVING_UTILIZATION_PCT AS UTIL_PCT,
            NUM_DELINQ_30_12M AS DELINQ_30,
            ROUND(PREDICTION:output_feature_1::FLOAT, 3) AS PROB_DEFAULT,
            CASE WHEN PREDICTION:output_feature_1::FLOAT > 0.35 THEN 'DECLINE' ELSE 'APPROVE' END AS DECISION,
            ACTUAL_DEFAULT
        FROM scored
        ORDER BY PROB_DEFAULT DESC
        """
        df = session.sql(batch_query).to_pandas()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Scored", len(df))
        with c2:
            approve_rate = (df["DECISION"] == "APPROVE").mean()
            st.metric("Approval Rate", f"{approve_rate:.1%}")
        with c3:
            decline_defaults = df[(df["DECISION"] == "DECLINE") & (df["ACTUAL_DEFAULT"] == 1)]
            if len(df[df["DECISION"] == "DECLINE"]) > 0:
                precision = len(decline_defaults) / len(df[df["DECISION"] == "DECLINE"])
                st.metric("Decline Precision", f"{precision:.1%}", help="% of declines that actually defaulted")
        
        st.dataframe(df, use_container_width=True)
