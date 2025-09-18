from src.hardware_monitor import log_hardware_info
from src.data_loader import load_data
from src.eda_analysis import run_eda
from src.preprocessing import preprocess_data
from src.model_setup import get_models, get_param_grids
from src.model_training import train_and_evaluate


if __name__ == "__main__":
    hardware_info, hardware_file = log_hardware_info()
    df = load_data()
    num_features, cat_features = run_eda(df, pdf_path="figures/eda_report.pdf")
    X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor = preprocess_data(df, num_features, cat_features)
    
    # Compute pos_weight dynamically from training data
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Load models and parameter grids
    models = get_models(pos_weight)
    param_grids = get_param_grids()


    best_models, results_val, cv_results, runtime_results, overfitting_analysis, resource_usage = train_and_evaluate(
        models, param_grids, preprocessor, X_train, y_train, X_valid, y_valid, pp
    )
