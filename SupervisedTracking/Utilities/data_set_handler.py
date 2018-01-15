from random import randint
import numpy as np


class DataSetHandler(object):
    def __init__(self, dataset_dir_path, open_dataset, label_creator, pre_processing):
        self.dataset_dir_path = dataset_dir_path
        self.dataset = open_dataset(dataset_dir_path)
        self.label_creator = label_creator
        self.pre_processing = pre_processing
        self.iterator = 0

    def next_batch(self, batch_size, start_index=None, random_batch=False, pre_processing=True):
        x_batch = []
        y_batch = []
        if start_index is not None:
            for idx in range(start_index, start_index + batch_size):
                x, y, metadata = self.label_creator(self.dataset.get(idx % self.dataset.len()))
                if pre_processing:
                    x, y = self.pre_processing(x, y)
                x_batch.append(x)
                y_batch.append(y)
        elif random_batch:
            for _ in range(batch_size):
                im_index = randint(0, self.dataset.len())
                x, y, metadata = self.label_creator(self.dataset.get(im_index))
                if pre_processing:
                    x, y = self.pre_processing(x, y)
                x_batch.append(x)
                y_batch.append(y)
        else:
            for _ in range(batch_size):
                x, y, metadata = self.label_creator(self.dataset.get(self.iterator % self.dataset.len()))
                if pre_processing:
                    x, y = self.pre_processing(x, y)
                x_batch.append(x)
                y_batch.append(y)
                self.iterator += 1
        return np.array(x_batch), np.array(y_batch), metadata

    def len(self):
        return self.dataset.len()

    # def dispose(self):
    #     for path in self.dataset:
    #         delete_file(path)


class Dataset(object):
    def __init__(self, data): # data - list of lists of data, assumes that all lists from the same size
        self.data = data
        self.length = len(data[0])
        self.datasets_amount = len(data)

    def get(self, index):
        res = []
        for i in range(self.datasets_amount):
            res.append(self.data[i][index])
        return res

    def len(self):
        return self.length


