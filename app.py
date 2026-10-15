import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Failure Prediction System",
    page_icon="⚙️",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

DATA_PATH = "machine_failure_prediction_dataset.csv"

MODEL_PATH = "models/best_model.pkl"

RESULT_PATH = "models/model_comparison.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(DATA_PATH):

    st.error(
        "❌ Dataset file not found!"
    )

    st.info(
        "Make sure machine_failure_prediction_dataset.csv "
        "is in the same folder as app.py"
    )

    st.stop()


if not os.path.exists(MODEL_PATH):

    st.error(
        "❌ Trained model not found!"
    )

    st.info(
        "Run all cells in the Jupyter Notebook first."
    )

    st.stop()


# =========================================================
# LOAD
# =========================================================

df = load_data()

model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("⚙️ Machine Failure Prediction System")

st.markdown(
    """
    **AI-Based Predictive Maintenance System**

    This application predicts the probability of machine failure
    using machine sensor parameters and Machine Learning.
    """
)


st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "🤖 Machine Prediction",
        "📊 Data Analysis",
        "📈 Model Performance"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header("🏭 Factory Overview")

    # Basic calculations

    total_records = len(df)

    total_machines = df["Machine_ID"].nunique()

    total_failures = int(
        df["Machine_Failure"].sum()
    )

    failure_rate = (
        total_failures / total_records
    ) * 100


    # Metrics

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Machines",
            total_machines
        )


    with col2:

        st.metric(
            "Sensor Records",
            f"{total_records:,}"
        )


    with col3:

        st.metric(
            "Total Failures",
            total_failures
        )


    with col4:

        st.metric(
            "Failure Rate",
            f"{failure_rate:.2f}%"
        )


    st.divider()


    # Machine Type Distribution

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Machine Type Distribution"
        )

        machine_count = (
            df["Machine_Type"]
            .value_counts()
            .reset_index()
        )

        machine_count.columns = [
            "Machine_Type",
            "Count"
        ]


        fig = px.pie(
            machine_count,
            names="Machine_Type",
            values="Count",
            hole=0.4
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Failure Rate by Machine Type"
        )

        failure_type = (
            df.groupby("Machine_Type")
            ["Machine_Failure"]
            .mean()
            .reset_index()
        )


        failure_type["Failure_Rate"] = (
            failure_type["Machine_Failure"] * 100
        )


        fig = px.bar(
            failure_type,
            x="Machine_Type",
            y="Failure_Rate",
            text="Failure_Rate",
            labels={
                "Failure_Rate":
                "Failure Rate (%)"
            }
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Sensor Overview

    st.subheader(
        "📡 Sensor Overview"
    )


    sensor = st.selectbox(
        "Select Sensor",
        [
            "Temperature_C",
            "Vibration_mm_s",
            "RPM",
            "Torque_Nm",
            "Tool_Wear_%",
            "Air_Pressure_bar",
            "Operating_Hours",
            "Power_Consumption_kW"
        ]
    )


    fig = px.histogram(
        df,
        x=sensor,
        color="Machine_Failure",
        nbins=40,
        title=f"{sensor} Distribution"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MACHINE PREDICTION
# =========================================================

elif page == "🤖 Machine Prediction":

    st.header(
        "🤖 Machine Failure Prediction"
    )


    st.write(
        "Enter the current machine sensor readings "
        "to predict failure risk."
    )


    st.divider()


    # Machine information

    machine_id = st.text_input(
        "Machine ID",
        value="M001"
    )


    machine_type = st.selectbox(
        "Machine Type",
        [
            "CNC",
            "Lathe",
            "Milling",
            "Drilling"
        ]
    )


    st.subheader(
        "📡 Machine Sensor Parameters"
    )


    col1, col2 = st.columns(2)


    # LEFT COLUMN

    with col1:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            max_value=150.0,
            value=70.0,
            step=0.1
        )


        vibration = st.number_input(
            "Vibration (mm/s)",
            min_value=0.0,
            max_value=20.0,
            value=3.5,
            step=0.1
        )


        rpm = st.number_input(
            "RPM",
            min_value=0,
            max_value=5000,
            value=2200,
            step=50
        )


        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=150.0,
            value=55.0,
            step=1.0
        )


    # RIGHT COLUMN

    with col2:

        tool_wear = st.number_input(
            "Tool Wear (%)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=1.0
        )


        air_pressure = st.number_input(
            "Air Pressure (bar)",
            min_value=0.0,
            max_value=10.0,
            value=6.2,
            step=0.1
        )


        operating_hours = st.number_input(
            "Operating Hours",
            min_value=0,
            max_value=10000,
            value=1800,
            step=50
        )


        power_consumption = st.number_input(
            "Power Consumption (kW)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.1
        )


    st.divider()


    # PREDICT BUTTON

    predict_button = st.button(
        "🔮 PREDICT MACHINE FAILURE",
        use_container_width=True
    )


    if predict_button:

        # =================================================
        # CREATE INPUT DATA
        # =================================================

        input_data = pd.DataFrame({

            "Machine_Type": [
                machine_type
            ],

            "Temperature_C": [
                temperature
            ],

            "Vibration_mm_s": [
                vibration
            ],

            "RPM": [
                rpm
            ],

            "Torque_Nm": [
                torque
            ],

            "Tool_Wear_%": [
                tool_wear
            ],

            "Air_Pressure_bar": [
                air_pressure
            ],

            "Operating_Hours": [
                operating_hours
            ],

            "Power_Consumption_kW": [
                power_consumption
            ]

        })


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            input_data
        )[0]


        probability = (
            model.predict_proba(
                input_data
            )[0][1] * 100
        )


        st.divider()

        st.subheader(
            "📋 Prediction Result"
        )


        # =================================================
        # RESULT
        # =================================================

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Machine",
                machine_id
            )


        with col2:

            st.metric(
                "Failure Probability",
                f"{probability:.2f}%"
            )


        with col3:

            if probability >= 75:

                risk = "HIGH"

            elif probability >= 40:

                risk = "MEDIUM"

            else:

                risk = "LOW"


            st.metric(
                "Risk Level",
                risk
            )


        st.divider()


        # =================================================
        # STATUS
        # =================================================

        if probability >= 75:

            st.error(
                "🔴 HIGH FAILURE RISK"
            )

            st.warning(
                "Immediate preventive maintenance is recommended."
            )


        elif probability >= 40:

            st.warning(
                "🟠 MEDIUM FAILURE RISK"
            )

            st.info(
                "Monitor the machine condition closely."
            )


        else:

            st.success(
                "🟢 LOW FAILURE RISK"
            )

            st.success(
                "Machine is currently operating normally."
            )


        # =================================================
        # GAUGE
        # =================================================

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={
                    "text":
                    "Machine Failure Probability"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    }
                }
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # =================================================
        # MAINTENANCE RECOMMENDATION
        # =================================================

        st.subheader(
            "🛠️ Maintenance Recommendation"
        )


        if probability >= 75:

            st.error(
                """
                **Immediate Action Required**

                • Stop or inspect the machine if operationally appropriate  
                • Check vibration levels  
                • Inspect tool wear  
                • Check temperature  
                • Schedule preventive maintenance
                """
            )


        elif probability >= 40:

            st.warning(
                """
                **Monitoring Required**

                • Monitor vibration  
                • Check temperature trend  
                • Inspect tool wear  
                • Review operating hours  
                • Plan maintenance inspection
                """
            )


        else:

            st.success(
                """
                **Normal Operation**

                • Continue regular monitoring  
                • Follow scheduled maintenance  
                • Record sensor readings regularly
                """
            )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "📊 Data Analysis":

    st.header(
        "📊 Machine Sensor Data Analysis"
    )


    st.subheader(
        "Dataset Preview"
    )


    st.dataframe(
        df.head(100),
        use_container_width=True
    )


    st.subheader(
        "Dataset Statistics"
    )


    st.dataframe(
        df.describe(),
        use_container_width=True
    )


    st.divider()


    # Correlation

    st.subheader(
        "🔥 Sensor Correlation"
    )


    numeric_data = df.select_dtypes(
        include=np.number
    )


    correlation = numeric_data.corr()


    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Scatter plot

    st.subheader(
        "📈 Sensor Relationship"
    )


    x_axis = st.selectbox(
        "X Axis",
        [
            "Temperature_C",
            "Vibration_mm_s",
            "RPM",
            "Torque_Nm",
            "Tool_Wear_%",
            "Air_Pressure_bar",
            "Operating_Hours",
            "Power_Consumption_kW"
        ]
    )


    y_axis = st.selectbox(
        "Y Axis",
        [
            "Temperature_C",
            "Vibration_mm_s",
            "RPM",
            "Torque_Nm",
            "Tool_Wear_%",
            "Air_Pressure_bar",
            "Operating_Hours",
            "Power_Consumption_kW"
        ],
        index=1
    )


    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="Machine_Failure",
        hover_data=[
            "Machine_ID",
            "Machine_Type"
        ],
        title=f"{x_axis} vs {y_axis}"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.header(
        "📈 Machine Learning Model Performance"
    )


    if os.path.exists(RESULT_PATH):

        results = pd.read_csv(
            RESULT_PATH
        )


        st.subheader(
            "Model Comparison"
        )


        # Percentage display copy

        display_results = results.copy()

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC"
        ]


        for column in metric_columns:

            display_results[column] = (
                display_results[column] * 100
            ).round(2)


        st.dataframe(
            display_results,
            use_container_width=True
        )


        st.divider()


        # Accuracy

        st.subheader(
            "Accuracy Comparison"
        )


        fig = px.bar(
            display_results,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            title="Model Accuracy"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Precision

        st.subheader(
            "Precision Comparison"
        )


        fig = px.bar(
            display_results,
            x="Model",
            y="Precision",
            text="Precision",
            title="Model Precision"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Recall

        st.subheader(
            "Recall Comparison"
        )


        fig = px.bar(
            display_results,
            x="Model",
            y="Recall",
            text="Recall",
            title="Model Recall"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # F1 Score

        st.subheader(
            "F1 Score Comparison"
        )


        fig = px.bar(
            display_results,
            x="Model",
            y="F1_Score",
            text="F1_Score",
            title="Model F1 Score"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # Best model

        best_model_row = results.loc[
            results["F1_Score"].idxmax()
        ]


        st.success(
            f"🏆 Best Model: {best_model_row['Model']}"
        )


    else:

        st.warning(
            "Model comparison file not found."
        )

        st.info(
            "Run all cells in the Jupyter Notebook first."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Machine Failure Prediction System | "
    "Python • Machine Learning • XGBoost • Streamlit"
)