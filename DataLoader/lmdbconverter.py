# lmdbconverter.py

import os
import cv2
import fire
import glob
import lmdb
import logging
import pyarrow
import lz4framed
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
import jpeg4py as jpeg
from itertools import tee
from typing import Generator, Any


logging.basicConfig(level=logging.INFO,
                    format= '[%(asctime)s] [%(pathname)s:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
DATA_DIRECTORY = './data'
IMAGE_NAMES_FILE = 'image_names.csv'


def list_files_in_folder(folder_path: str) -> Generator:
    return (file_name__str for file_name__str in glob.glob(os.path.join(folder_path, "*.*")))


def read_image_safely(image_file_name: str) -> np.array:
    try:
        return jpeg.JPEG(image_file_name).decode().astype(np.uint8)
    except jpeg.JPEGRuntimeError:
        return np.array([], dtype=np.uint8)


def serialize_and_compress(obj: Any):
    return lz4framed.compress(pyarrow.serialize(obj).to_buffer())


def extract_image_name(image_path: str) -> str:
    return image_path.split('/').pop(-1)


def resize(image_array, size=(256, 256)):
    if image_array.size == 0:
        return image_array
    return cv2.resize(image_array, dsize=size, interpolation=cv2.INTER_CUBIC)


def convert(image_folder: str, lmdb_output_path: str, write_freq: int=5000):
    assert os.path.isdir(image_folder), f"Image folder '{image_folder}' does not exist"
    assert not os.path.isfile(lmdb_output_path), f"LMDB store '{lmdb_output_path} already exists"
    assert not os.path.isdir(lmdb_output_path), f"LMDB store name should a file, found directory: {lmdb_output_path}"
    assert write_freq > 0, f"Write frequency should be a positive number, found {write_freq}"

    logger.info(f"Creating LMDB store: {lmdb_output_path}")

    image_file: Generator = list_files_in_folder(image_folder)
    image_file, image_file__iter_c1, image_file__iter_c2, image_file__iter_c3 = tee(image_file, 4)

    img_path_img_array__tuples = map(lambda tup: (tup[0], resize(read_image_safely(tup[1]))),
                                     zip(image_file__iter_c1, image_file__iter_c2))

    lmdb_connection = lmdb.open(lmdb_output_path, subdir=False,
                                map_size=int(4e11), readonly=False,
                                meminit=False, map_async=True)

    lmdb_txn = lmdb_connection.begin(write=True)
    total_records = 0

    try:
        for idx, (img_path, img_arr) in enumerate(tqdm(img_path_img_array__tuples)):
            img_idx: bytes = u"{}".format(idx).encode('ascii')
            img_name: str = extract_image_name(image_path=img_path)
            img_name: bytes = u"{}".format(img_name).encode('ascii')
            if idx < 5:
                logger.debug(img_idx, img_name, img_arr.size, img_arr.shape)
            lmdb_txn.put(img_idx, serialize_and_compress((img_name, img_arr.tobytes(), img_arr.shape)))
            total_records += 1
            if idx % write_freq == 0:
                lmdb_txn.commit()
                lmdb_txn = lmdb_connection.begin(write=True)
    except TypeError:
        logger.error(traceback.format_exc())
        lmdb_connection.close()
        raise

    lmdb_txn.commit()

    logger.info("Finished writing image data. Total records: {}".format(total_records))

    logger.info("Writing store metadata")
    image_keys__list = [u'{}'.format(k).encode('ascii') for k in range(total_records)]
    with lmdb_connection.begin(write=True) as lmdb_txn:
        lmdb_txn.put(b'__keys__', serialize_and_compress(image_keys__list))

    logger.info("Flushing data buffers to disk")
    lmdb_connection.sync()
    lmdb_connection.close()

    # -- store the order in which files were inserted into LMDB store -- #
    pd.Series(image_file__iter_c3).apply(extract_image_name).to_csv(os.path.join(DATA_DIRECTORY, IMAGE_NAMES_FILE),
                                                                    index=False, header=False)
    logger.info("Finished creating LMDB store")


if __name__ == '__main__':
    fire.Fire(convert)
    
'''
python3 lmdbconverter.py --image_folder ./images/ --lmdb_output_path ./data/lmdb-store.db
'''