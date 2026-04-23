from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- load training data ------------------------------
    train_data_obj = Dataset_Loader('train data', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'
    train_data = train_data_obj.load()

    # ---- load testing data -------------------------------
    test_data_obj = Dataset_Loader('test data', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'
    test_data = test_data_obj.load()

    # ---- method initialization ---------------------------
    method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.data = {
        'train': train_data,
        'test': test_data
    }

    # ---- running section --------------------------------
    print('************ Start ************')
    result = method_obj.run()

    # ---- evaluation section ------------------------------
    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    evaluate_obj.data = {
        'true_y': result['true_y'],
        'pred_y': result['pred_y']
    }

    acc = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('MLP Accuracy:', acc)
    print('************ Finish ************')


    