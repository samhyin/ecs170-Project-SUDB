'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    train_dataset_source_file_name = None
    test_dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load_one_file(self, file_name):
        X = []
        y = []
        file_path = os.path.join(self.dataset_source_folder_path, file_name)
        f = open(file_path, 'r', encoding='utf-8')
        for line in f:
            line = line.strip()
            if not line:
                continue
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}

    def load(self):
        print('loading data...')

        train_file_name = self.train_dataset_source_file_name or self.dataset_source_file_name
        test_file_name = self.test_dataset_source_file_name

        if test_file_name is None:
            return self.load_one_file(train_file_name)

        train_data = self.load_one_file(train_file_name)
        test_data = self.load_one_file(test_file_name)
        return {'train': train_data, 'test': test_data}
