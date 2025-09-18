from src.hardware_monitor import log_hardware_info
from src.data_loader import load_data
from src.eda_analysis import run_eda

if __name__ == "__main__":
    hardware_info, hardware_file = log_hardware_info()
    df = load_data()
    num_features, cat_features = run_eda(df, pdf_path="figures/eda_report.pdf")

