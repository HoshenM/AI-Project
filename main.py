from src.hardware_monitor import log_hardware_info
from src.data_loader import load_data
from src.eda_analysis import run_eda
from src.preprocessing import preprocess_data

if __name__ == "__main__":
    hardware_info, hardware_file = log_hardware_info()
    df = load_data()
    num_features, cat_features = run_eda(df, pdf_path="figures/eda_report.pdf")
    X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor = preprocess_data(df, num_features, cat_features)

