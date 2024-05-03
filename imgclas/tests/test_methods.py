"""
Gather all module's test

Date: December 2019
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""
import glob
import json
import os
import shutil
import subprocess
import time
from urllib.parse import quote_plus

from imgclas import paths


module_name = 'imgclas'
test_url = 'https://file-examples.com/storage/fe4996602366316ffa06467/2017/10/file_example_JPG_100kB.jpg'

data_path = os.path.join(paths.get_base_dir(), 'data')


def copy_files(src, dst, extension):
    files = glob.iglob(os.path.join(src, extension))
    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, dst)


def remove_files(src, extension):
    files = glob.iglob(os.path.join(src, extension))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)

def test_load():
    print('Testing local: module load ...')
    import imgclas.api


def test_metadata():
    print('Testing local: metadata ...')
    from imgclas.api import get_metadata

    get_metadata()


def test_predict_url():
    print('Testing local: predict url ...')
    from imgclas.api import predict_url

    args = {'urls': [test_url]}
    r = predict_url(args)


def test_predict_data():
    print('Testing local: predict data ...')
    from deepaas.model.v2.wrapper import UploadedFile
    from imgclas.api import predict_data

    fpath = os.path.join(data_path, 'samples', 'sample.jpg')
    tmp_fpath = os.path.join(data_path, 'samples', 'tmp_file.jpg')
    shutil.copyfile(fpath, tmp_fpath)  # copy to tmp because we are deleting the file after prediction
    file = UploadedFile(name='data', filename=tmp_fpath, content_type='image/jpg')
    args = {'files': [file]}
    r = predict_data(args)


def test_train():
    print('Testing local: train ...')

    from imgclas.api import get_train_args, train

    copy_files(src=os.path.join(data_path, 'demo-dataset_files'),
               dst=os.path.join(data_path, 'dataset_files'),
               extension='*.txt')

    args = get_train_args()
    args_d = {k: v.missing for k, v in args.items()}
    args_d['images_directory'] = '"data/samples"'
    args_d['modelname'] = '"MobileNet"'
    args_d['use_multiprocessing'] = 'false'
    out = train(**args_d)

    remove_files(src=os.path.join(data_path, 'dataset_files'),
                 extension='*.txt')

    # shutil.rmtree(os.path.join(paths.get_models_dir(), out['modelname']), ignore_errors=True)


if __name__ == '__main__':
    test_load()
    # test_metadata()
    # test_predict_url()
    # test_predict_data()

    # Train function cannot be run in the same loop as predict due to very poor usage of the config file in api.py
    # test_train()
