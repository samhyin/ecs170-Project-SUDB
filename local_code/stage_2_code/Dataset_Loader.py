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
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        # f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'r')
        file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        f = open(file_path, 'r')
        for line in f:
            line = line.strip('\n')
            if not line: continue
            elements = [int(i) for i in line.split(',')]
            X.append(elements[1:])
            y.append(elements[0])
        f.close()
        return {'X': X, 'y': y}