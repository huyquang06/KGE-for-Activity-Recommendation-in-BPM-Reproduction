import os
import shutil

def copy_files_with_patterns(source_folders, target_folders, file_patterns):
    for source, target in zip(source_folders, target_folders):
        # Ensure source and target folders exist
        if not os.path.exists(source):
            print(f"Source folder '{source}' does not exist. Skipping.")
            continue
        if not os.path.exists(target):
            os.makedirs(target)

        # Get a list of files in the source folder that match the specified patterns
        files_to_copy = [file for file in os.listdir(source) if any(pattern in file for pattern in file_patterns)]

        # Copy each matching file to the target folder
        for file_name in files_to_copy:
            source_path = os.path.join(source, file_name)
            target_path = os.path.join(target, file_name)
            shutil.copy2(source_path, target_path)
            #print(f"Copied '{file_name}'.")

if __name__ == "__main__":
    # Get the path to the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to the bpmai datasets folder
    bpmai_path = os.path.join(script_dir, "bpmai datasets")
    configs_path = os.path.join(script_dir, "configs")
    # Construct the path to the "data" folder
    local_folder_path = os.path.join(script_dir, "..")
    kge_folder_path = os.path.join(local_folder_path, "..")
    data_folder_path = os.path.join(kge_folder_path, "data")
    preprocess_folder_path = os.path.join(data_folder_path, "preprocess")
    # define dataset names
    dataset_names = ["bpmai_lastrev_caise_onlyAfter", "bpmai_lastrev_caise_onlyAfter2",
                     "bpmai_lastrev_caise_inProcess", "bpmai_lastrev_caise_inProcess2",
                     "bpmai_lastrev_caise", "bpmai_lastrev_caise2"]
    # Specify the file patterns for copy
    file_patterns = [".txt", ".yaml"]
    
    # Transfer all dataset files 
    source_folders, target_folders = [], []
    for i in range (6):
        source_folders.append(os.path.join(bpmai_path, dataset_names[i]))
        target_folders.append(os.path.join(data_folder_path, dataset_names[i]))

    copy_files_with_patterns(source_folders, target_folders, file_patterns)
    
    # Transfer all configuration files 
    source_folders, target_folders = [], []
    for i in range (6):
        source_folders.append(os.path.join(configs_path, dataset_names[i]))
        target_folders.append(os.path.join(data_folder_path, dataset_names[i]))

    copy_files_with_patterns(source_folders, target_folders, file_patterns)
    
    # copy default preprocess file to handle preprocessing of datasets smoothly
    preprocess_file = "preprocess_default.py"
    preprocess_source_path = os.path.join(preprocess_folder_path, preprocess_file)
    preprocess_destination_path = os.path.join(data_folder_path, preprocess_file)
    shutil.copy(preprocess_source_path, preprocess_destination_path)
    #print(f"File copied from {preprocess_source_path} to {preprocess_destination_path}")
    
    # copy two pyton files to smoothly handle performance evaluation
    kge_evaluation_file = "kge_evaluation.py"
    fileter_and_evaluate_file = "filter_and_evaluate.py"
    kge_evaluation_source_path = os.path.join(script_dir, kge_evaluation_file)
    kge_evaluation_destination_path = os.path.join(kge_folder_path, kge_evaluation_file)
    fileter_and_evaluate_source_path = os.path.join(script_dir, fileter_and_evaluate_file)
    fileter_and_evaluate_destination_path = os.path.join(kge_folder_path, fileter_and_evaluate_file)
    shutil.copy(kge_evaluation_source_path, kge_evaluation_destination_path)
    shutil.copy(fileter_and_evaluate_source_path, fileter_and_evaluate_destination_path)
    
    