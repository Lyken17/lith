import lmdb

import torch
import torch.utils.data as data


class LMDB(data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __add__(self, other):
        pass


class Cached(data.Dataset):
    def __init__(self, dst, cache_type="lmdb", target_dir=None):
        self.dst = dst

        self.current_dst = LMDB(target_dir)
        self.cached_lst = []

    def __getitem__(self, index):
        if index in self.cached_lst:
            return self.current_dst[index]

        res = self.dst[index]
        self.current_dst.insert(index, res)
        self.cached_lst.append(index)
        return res

    def __len__(self):
        return len(self.dst)

    def __add__(self, other):
        raise NotImplementedError("Concat of two datasets not supported for Cached dataset")
