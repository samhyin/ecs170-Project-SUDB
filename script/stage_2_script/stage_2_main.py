from Dataset_Loader import Dataset_Loader
from Method_MLP import Method_MLP
from Evaluate_Accuracy import Evaluate_Accuracy # 刚才改名的那个

# 加载训练集
train_loader = Dataset_Loader()
train_loader.dataset_source_folder_path = 'D:\Document\Program\ECS170_Spring_2026_Source_Code_Template\data\stage_2_data' # 你的路径
train_loader.dataset_source_file_name = 'train.csv'
train_data = train_loader.load()

# 加载测试集
test_loader = Dataset_Loader()
test_loader.dataset_source_folder_path = 'D:\Document\Program\ECS170_Spring_2026_Source_Code_Template\data\stage_2_data'
test_loader.dataset_source_file_name = 'test.csv'
test_data = test_loader.load()

# 初始化模型
mlp_model = Method_MLP('MLP', 'Multi-layer Perceptron for MNIST')
mlp_model.data = {'train': train_data, 'test': test_data}

# 运行与评估
result = mlp_model.run()
evaluator = Evaluate_Accuracy()
evaluator.data = result
evaluator.evaluate()