from os.path import join
import numpy as np
import h5py
import torch

seq_length = 4

class Dataset(object):
    def __init__(self, root, ids, n,
                 max_examples=None, bound=10, cache=False):
        self.ids = list(ids)
        self.n = n
        self.bound = bound

        if max_examples is not None:
            self.ids = self.ids[:max_examples]

        self.data = h5py.File(join(root, 'data.hdf5'), 'r')

        self.cache = None
        if cache:
            self.cache = []
            for id in ids:
                self.cache.append(self.single_item(id))

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]

        return self.single_item(self.ids[idx])

    def single_item(self, id):
        if isinstance(id, bytes):
            id = id.decode("utf-8")

        image = self.data[id]['image'][()]/255.*2 - 1
        image = torch.Tensor(image.transpose(2, 0, 1)).unsqueeze(0)
        pose = torch.Tensor(self.data[id]['pose'][()]).unsqueeze(0)

        enough = False
        id_num = int(id[-seq_length:])
        while not enough:
            random_num = np.random.randint(-self.bound, self.bound)
            id_target = id[:-seq_length] + str(id_num + random_num).zfill(seq_length)

            if id_target in self.data:
                image_tmp = self.data[id_target]['image'][()]/255.*2 - 1
                image_tmp = torch.Tensor(image_tmp.transpose(2, 0, 1)).unsqueeze(0)
                pose_tmp = torch.Tensor(self.data[id_target]['pose'][()]).unsqueeze(0)
                image = torch.cat((image, image_tmp), dim=0)
                pose = torch.cat((pose, pose_tmp), dim=0)

                if pose.shape[0] == self.n + 1:
                    enough = True

        return (image[1:], pose[1:]), (image[0], pose[0])

    def __len__(self):
        return len(self.ids)

def create_default_splits(n, root, bound=10, cache=False):
    ids_train = []
    ids_test = []
    with open(join(root, 'id_train.txt'), 'r') as fp:
        ids_train = [s.strip() for s in fp.readlines() if s]
    with open(join(root, 'id_test.txt'), 'r') as fp:
        ids_test = [s.strip() for s in fp.readlines() if s]

    dataset_train = Dataset(root, ids_train, n, bound=bound, cache=cache)
    dataset_test = Dataset(root, ids_test, n, bound=bound, cache=cache)

    return dataset_train, dataset_test
