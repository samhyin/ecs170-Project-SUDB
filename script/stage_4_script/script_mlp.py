import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import ensure_result_dir, resolve_stage_data_dir
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Method_MLP import Method_MLP
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test import Setting_Train_Test


def main():
    np.random.seed(2)
    torch.manual_seed(2)

    data_dir = resolve_stage_data_dir('stage_4_data', ('train.csv', 'test.csv'))
    result_dir = ensure_result_dir('stage_4_result')

    data_obj = Dataset_Loader('stage 4 inherited MLP template', 'Pre-split train/test dataset template')
    data_obj.dataset_source_folder_path = str(data_dir)
    data_obj.train_dataset_source_file_name = 'train.csv'
    data_obj.test_dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', 'Stage 4 inherited multiclass MLP template')
    method_obj.curve_output_path = result_dir / 'stage_4_loss_curve.png'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = str(result_dir / 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('pre-partitioned train/test', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', 'Macro precision/recall/F1 included')

    print('************ Start ************')
    print(f'using stage 4 data directory: {data_dir}')
    print(f'using stage 4 result directory: {result_dir}')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    if std_score is None:
        print('MLP Accuracy: ' + str(mean_score))
    else:
        print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('saved convergence curve to:', method_obj.curve_output_path)
    print(
        'saved prediction result to:',
        result_obj.result_destination_folder_path
        + result_obj.result_destination_file_name
        + '_'
        + str(result_obj.fold_count),
    )
    print('************ Finish ************')


if __name__ == '__main__':
    main()
