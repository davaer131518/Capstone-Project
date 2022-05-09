import os
import uuid
import shutil
import logging
import numpy as np
from tqdm import tqdm
from typing import Union, List, NoReturn
from config import DATA_DIR, ReturnCodes, data_class2id_map, data_id2class_map


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_data(split_ratio: float = 0.1) -> Union[List, List]:

    testing_paths = []
    training_paths = []

    for label_path in os.listdir(DATA_DIR):
        if 'train' in  label_path or 'test' in label_path:
            continue
        else:
            
            sub_path = os.path.join(DATA_DIR, label_path)
            
            label_id = data_class2id_map[label_path]
            image_type_path = os.path.join(DATA_DIR,label_path)
            files =  os.listdir(image_type_path)
            num_images = len(files)
            test_images = list(np.random.choice(files, size=int(num_images*split_ratio), replace=False))
            train_images = list(set(files) - set(test_images))

            assert sorted(train_images + test_images) == sorted(files)
            
            testing_paths += [os.path.join(sub_path, t_i) for t_i in test_images]
            training_paths += [os.path.join(sub_path, t_i) for t_i in train_images]

    return training_paths, testing_paths
    

def make_train_test() -> ReturnCodes:
    
    finish_code = ReturnCodes.FAIL_WITH_ERROR
    try:
        [os.makedirs(os.path.join(DATA_DIR, 'train', folder), exist_ok=True) for folder in list(data_class2id_map.keys())]
        [os.makedirs(os.path.join(DATA_DIR, 'test', folder), exist_ok=True) for folder in list(data_class2id_map.keys())]

        finish_code = ReturnCodes.SUCCESS
    except RuntimeError as e:
        print("Unable to relable the data witrh error: ", e)
        return finish_code
    return finish_code

def relabel_data() -> ReturnCodes:
    finish_code = ReturnCodes.FAIL_WITH_ERROR
    try:

        for label_path in tqdm(os.listdir(DATA_DIR)):
            if 'train' in  label_path or 'test' in label_path:
                continue
                
            logger.info(label_path)

            label_id = data_class2id_map[label_path]
            image_type_path = os.path.join(DATA_DIR,label_path)
            for image_path in os.listdir(image_type_path):
                image_id = uuid.uuid4().int
                image_name = f"{image_id}_{label_id}.jpg"
                os.rename(os.path.join(image_type_path, image_path), os.path.join(image_type_path,image_name))
            logger.info("_______________________")

        finish_code = ReturnCodes.SUCCESS
    except RuntimeError as e:
        print("Unable to relable the data witrh error: ", e)
        return finish_code
    return finish_code

def train_test_split_paths(data_paths: List = [], split_type:str = "train") -> ReturnCodes:
    finish_code = ReturnCodes.FAIL_WITH_ERROR
    try:
        for data_path in data_paths:
            output_path = os.path.join(DATA_DIR, split_type)
            data_label_folder = data_id2class_map[int(data_path.split('_')[-1][0])]
            output_path = os.path.join(output_path, data_label_folder)
            output_path = os.path.join(output_path, os.path.basename(data_path))
            
            shutil.copyfile(data_path, output_path)
        
        finish_code = ReturnCodes.SUCCESS

    except RuntimeError as e:
        print("Unable to copy data with error: ", e)
        return finish_code
    return finish_code



if __name__ == "__main__":
    relabel_data()
    make_train_test()
    training_paths, testing_paths = make_data()

    finish_code = train_test_split_paths(training_paths, 'train')
    finish_code = train_test_split_paths(testing_paths, 'test')