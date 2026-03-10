# BTCUSD RNN Time Series Forecasting Project Plan

This project aims to build a production-ready Machine Learning pipeline for forecasting BTCUSD prices using Recurrent Neural Networks (RNNs). It incorporates MLOps best practices, including experiment tracking, model registry, CI/CD, drift detection, and business impact analysis through A/B testing and ROI.

## **Project Overview**

Forecasting BTCUSD prices using historical market data with deep learning models (LSTM/GRU). The focus is not just on prediction accuracy but on the end-to-end lifecycle of a machine learning model in a production environment.

## **Core Components**

### **1. Time Series Forecasting Models**
- **Linear Regression**: A simple baseline model predicting future price based on current market features (Close, Volume, Moving Averages).
- **ARIMA**: AutoRegressive Integrated Moving Average model for univariate time series forecasting.
- **RNN (Optional)**: LSTM/GRU architectures for capturing complex non-linear dependencies (advanced use case).
- **Features**: Historical OHLCV (Open, High, Low, Close, Volume), technical indicators (RSI, Moving Averages).
- **Objective**: Predict the next-period price or price movement direction.

### **2. MLflow Experiments & Model Registry**
- **Experiment Tracking**: Log hyperparameters, metrics (RMSE, MAE), and model artifacts for every run.
- **Model Registry**: Version control models, transition models through stages (Staging, Production, Archived), and manage model metadata.

### **3. CI/CD for ML (MLOps)**
- **Continuous Integration (CI)**: Automated testing of data preprocessing scripts, model training code, and unit tests for model architecture.
- **Continuous Deployment (CD)**: Automatically deploy the best-performing model to a staging or production environment when it passes validation tests.
- **Pipeline Tools**: GitHub Actions, GitLab CI, or Jenkins.

### **4. Drift Detection & Monitoring**
- **Data Drift**: Monitor changes in the distribution of input features (e.g., sudden spikes in volatility or volume).
- **Model Drift**: Detect degradation in model performance (e.g., increasing RMSE) over time.
- **Tools**: Evidently AI, Alibi Detect, or custom monitoring scripts integrated with Prometheus/Grafana.

### **5. Django Dashboard**
- **Visualization**: A central hub to visualize model performance, MLflow experiment summaries, and drift detection reports.
- **A/B Testing Interface**: Manage and monitor live A/B tests between different model versions.
- **ROI Tracking**: Real-time display of financial impact and ROI based on current market predictions.
- **Alerting**: Integrated dashboard alerts for data drift or performance drops.

### **6. A/B Testing & ROI Analysis**
- **A/B Testing**: Compare the performance of the current production model (Control) against a new challenger model (Treatment) in a live or simulated environment.
- **ROI Analysis**: Quantify the financial impact of the model's predictions (e.g., simulated trading profits, risk reduction) to justify the project's value.

---

## **Project Roadmap**

### **Phase 1: Foundation & Data Engineering**
- [ ] Set up project structure and environment.
- [ ] Implement data ingestion from BTCUSD APIs (e.g., Binance, Coinbase).
- [ ] Perform Exploratory Data Analysis (EDA) and feature engineering.

### **Phase 2: Modeling & MLflow Integration**
- [] Develop Linear Regression model (`src/models/linear_regression.py`).
- [] Develop ARIMA model (`src/models/arima_model.py`).
- [ ] (Optional) Develop RNN model (LSTM/GRU).
- [ ] Integrate MLflow for experiment tracking.
- [ ] Implement a basic training pipeline.

### **Phase 3: Django Dashboard & Model Registry**
- [ ] Set up MLflow Model Registry.
- [ ] Initialize Django project and set up basic dashboard UI.
- [ ] Create a deployment script (e.g., FastAPI/Django-integrated REST API).
- [ ] Implement basic CI/CD for model training and deployment.

### **Phase 4: Monitoring & Drift Detection**
- [ ] Implement data and model drift detection using Evidently AI.
- [ ] Integrate monitoring reports into the Django Dashboard.
- [ ] Set up automated alerts for performance degradation.

### **Phase 5: Business Impact & A/B Testing**
- [ ] Design and implement an A/B testing framework within Django.
- [ ] Develop ROI calculation logic and display on dashboard.
- [ ] Final project documentation and visualization.

---

## **Proposed Project Structure**

```text
├── dashboard/              # Django Dashboard
│   ├── templates/          # HTML templates
│   ├── static/             # CSS/JS files
│   ├── apps/               # Monitoring, A/B Testing, ROI apps
│   └── manage.py           # Django entry point
├── src/                    # ML logic
│   ├── data/               # Data ingestion & preprocessing
│   ├── features/           # Feature engineering
│   ├── models/             # Model architectures & training
│   │   ├── linear_regression.py  # Linear Regression implementation
│   │   ├── arima_model.py        # ARIMA implementation
│   │   └── train.py              # RNN/General training script
│   ├── monitoring/         # Drift detection logic
│   └── tests/              # ML unit & integration tests
├── .github/workflows/      # CI/CD pipelines
├── data/                   # Raw and processed data
├── mlflow/                 # MLflow tracking server (if local)
├── config.yaml             # Project configuration
├── requirements.txt        # Dependencies
└── README.md               # Project plan and documentation
```

---

## **Getting Started**

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd timeseries
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run MLflow server**:
    ```bash
    mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri ./mlruns --allowed-hosts "*" --cors-allowed-origins "*"
    ```
