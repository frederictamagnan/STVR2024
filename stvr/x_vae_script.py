import joblib
import os
datasets=["femto_booking_agilkia_v6","spree_5000_session_wo_responses_agilkia","teaming_execution","scanette_100043-steps"]
archs=["AE","VAE","DAAE"]



def clean_directory(directory_path, except_file):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename != except_file and filename.startswith("model"):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)

for dataset_name in datasets:
    for arch in archs:
        study = joblib.load("./data/studies/study_"+dataset_name+"_"+arch+".pkl")
        print(dataset_name,arch,study.best_trial.number)
        except_file='model_trial_'+str(study.best_trial.number)+'.pt'
        directory_path="./data/checkpoints/"+dataset_name+"/"+arch+"/"
        clean_directory(directory_path,except_file)
