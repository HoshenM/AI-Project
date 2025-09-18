from matplotlib.backends.backend_pdf import PdfPages
from src.hardware_monitor import log_hardware_info
from src.data_loader import load_data
from src.eda_analysis import run_eda
from src.preprocessing import preprocess_data
from src.model_setup import get_models, get_param_grids
from src.model_training import train_and_evaluate
from src.results_analysis import analyze_results
from src.feature_importance import plot_feature_importances
from src.ensemble_methods import build_voting_ensemble
from src.learning_curves import plot_learning_curves
from src.evaluation import evaluate_on_test_set
from src.statistical_tests import compare_models_statistically
from src.project_summary import summarize_project
from src.save_results import save_results
from src.validation_curves import run_validation_curves
from src.performance_analysis import compare_validation_test_performance
from src.feature_engineering import aggregate_feature_importance
from src.deployment import predict_stroke_probability
from src.final_report import print_final_report


if __name__ == "__main__":
    # Detect hardware
    hardware_info, hardware_file = log_hardware_info()
    
    # Load data
    df = load_data()
    
    # EDA
    num_features, cat_features = run_eda(df, pdf_path="figures/eda_report.pdf")
    
    # Preprocess
    X_train, X_valid, X_test, y_train, y_valid, y_test, preprocessor = preprocess_data(
        df, num_features, cat_features
    )
    
    # === Create PDF container for all plots ===
    with PdfPages("figures/stroke_project_report.pdf") as pp:
        
        # Compute pos_weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        # Models + params
        models = get_models(pos_weight)
        param_grids = get_param_grids()

        # Train & evaluate
        best_models, results_val, cv_results, runtime_results, overfitting_analysis, resource_usage = train_and_evaluate(
            models, param_grids, preprocessor, X_train, y_train, X_valid, y_valid, pp
        )

        # Results analysis
        results_df, cv_df, overfitting_df = analyze_results(
            results_val, cv_results, overfitting_analysis
        )

        # Feature importance
        importance_models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
        plot_feature_importances(best_models, importance_models, pp, top_n=15)

        # Ensemble
        best_models, voting_results, voting_auc = build_voting_ensemble(
            best_models, results_df, X_train, y_train, X_valid, y_valid
        )
        results_val.update(voting_results)

        # Learning curves
        best_model_name = results_df["AUC"].idxmax()
        best_model = best_models[best_model_name]
        plot_learning_curves(best_model, best_model_name, X_train, y_train, pdf_saver=pp)

        # Final test evaluation
        final_test_df = evaluate_on_test_set(
            best_models, best_model_name, X_test, y_test, pdf_saver=pp
        )

        # Statistical test
        top_3_models = results_df.nlargest(3, 'AUC').index
        if len(top_3_models) >= 2:
            model1, model2 = top_3_models[0], top_3_models[1]
            scores1 = cv_results[model1]['CV_Scores']
            scores2 = cv_results[model2]['CV_Scores']
            compare_models_statistically(scores1, scores2, model1, model2)


        # Project Summary 
        summarize_project(models, best_model_name, results_df, test_results,
                          runtime_results, overfitting_df, pp)

        save_results(best_model_final, best_model_name, df, X, y,
             X_train, X_valid, X_test,
             models, results_df, results_val, test_results,
             cv_results, overfitting_analysis, runtime_results)

        run_validation_curves(models, preprocessor, X_train, y_train, preprocessor)


        compare_validation_test_performance(results_val, test_results)

        aggregate_feature_importance(best_models)

        # Example deployment
        sample_patient = X.iloc[0].to_dict()
        prediction = predict_stroke_probability(sample_patient, best_model_final)
        print("Example prediction:", prediction)

        print_final_report(df, X, models, best_model_name, results_df,
                           test_results, runtime_results)

