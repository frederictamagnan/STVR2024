from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import joblib
import plotly.io as pio


datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
archs=["AE","VAE","DAAE"]
plot=False
for dataset_name in datasets:
    for arch in archs:
        
        study = joblib.load("./data/studies/study_"+dataset_name+"_"+arch+".pkl")
        print(dataset_name,arch,study.best_trial.params)
        if plot:
            fig=plot_optimization_history(study)
            pio.write_image(fig, "./results/plot/param_optim_h_"+dataset_name+"_"+arch+".png")
            fig=plot_intermediate_values(study)
            pio.write_image(fig, "./results/plot/intermediate_values"+dataset_name+"_"+arch+".png")
            # pio.write_image(fig, "./results/plot/param_contour_"+dataset_name+"_"+arch+".png")
            # fig=plot_param_importances(study)
            # pio.write_image(fig, "./results/plot/param_importances_"+dataset_name+"_"+arch+".png")
