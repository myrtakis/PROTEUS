class FileNames:
    default_folder = '../results'
    navigator_fname = 'navigator.json'
    detector_info_fname = 'detectors_info.json'
    train_test_indices_fname = 'train_test_indices.json'
    pseudo_samples_info = 'pseudo_samples_info.json'
    best_models_bench_fname = 'models_benchmark.json'
    best_model_fname = 'best_model.json'


class FileKeys:
    navigator_conf_path = 'config_path'
    navigator_original_dataset_path = 'original_dataset_path'
    navigator_original_data = 'original_data'
    navigator_pseudo_samples_key = 'pseudo_samples'
    navigator_pseudo_samples_hold_out_data_key = 'hold_out_dataset_path'
    navigator_pseudo_sample_dir_key = 'dir'
    navigator_pseudo_samples_data_path = 'dataset_path'
    navigator_pseudo_samples_num_key = 'pseudo_samples_num'
    navigator_detector_info_path = 'detector_info'
