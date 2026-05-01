import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_code.path_utils import ensure_result_dir
from local_code.stage_5_code.Result_Loader import Result_Loader


def main():
    result_dir = ensure_result_dir('stage_5_result')
    result_obj = Result_Loader('loader', '')
    result_obj.result_destination_folder_path = str(result_dir / 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = None
    result_obj.load()
    print('Fold:', result_obj.fold_count, ', Result keys:', list(result_obj.data.keys()))
    print('Prediction count:', len(result_obj.data['pred_y']))
    print('Ground-truth count:', len(result_obj.data['true_y']))
    print('First 10 predictions:', result_obj.data['pred_y'][:10])
    print('First 10 labels:', result_obj.data['true_y'][:10])


if __name__ == '__main__':
    main()
