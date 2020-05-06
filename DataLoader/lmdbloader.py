# lmdbloader.py

import os
import lmdb
import pyarrow
import lz4framed
import numpy as np
from typing import Any
import nonechucks as nc
from torch.utils.data import Dataset, DataLoader


class InvalidFileException(Exception):
    pass


class LMDBDataset(Dataset):
    def __init__(self, lmdb_store_path, transform=None):
        super().__init__()
        assert os.path.isfile(lmdb_store_path), f"LMDB store '{lmdb_store_path} does not exist"
        assert not os.path.isdir(lmdb_store_path), f"LMDB store name should a file, found directory: {lmdb_store_path}"

        self.lmdb_store_path = lmdb_store_path
        self.lmdb_connection = lmdb.open(lmdb_store_path,
                                         subdir=False, readonly=True, lock=False, readahead=False, meminit=False)

        with self.lmdb_connection.begin(write=False) as lmdb_txn:
            self.length = lmdb_txn.stat()['entries'] - 1
            self.keys = pyarrow.deserialize(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            print(f"Total records: {len(self.keys), self.length}")
        self.transform = transform

    def __getitem__(self, index):
        lmdb_value = None
        with self.lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[index])
        assert lmdb_value is not None, f"Read empty record for key: {self.keys[index]}"

        img_name, img_arr, img_shape = LMDBDataset.decompress_and_deserialize(lmdb_value=lmdb_value)
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        if image.size == 0:
            raise InvalidFileException("Invalid file found, skipping")
        return image

    @staticmethod
    def decompress_and_deserialize(lmdb_value: Any):
        return pyarrow.deserialize(lz4framed.decompress(lmdb_value))

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = nc.SafeDataset(LMDBDataset('./data/lmdb-tmp.db'))
    batch_size = 64
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=False)
    n_epochs = 50

    for _ in range(n_epochs):
        for batch in data_loader:
            assert len(batch) > 0