import os
from os.path import isfile, join, isdir
import errno
import shutil
import csv
import time
import threading
import json


def dir_to_file_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isfile(join(path_to_dir, f))]


def dir_to_file_list_with_ext(path_to_dir, ext):
    return list(filter(lambda x: x.endswith(ext), dir_to_file_list(path_to_dir)))


def dir_to_subdir_list(path_to_dir):
    return [path_to_dir + '/' + f for f in os.listdir(path_to_dir) if isdir(join(path_to_dir, f))]


def create_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def delete_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def delete_file(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def get_file_name(path):
    path = path.replace('\\', '/')
    return path.split('/')[-1]


def get_file_name_without_ext(path):
    name_with_ext = get_file_name(path)
    return name_with_ext.split(".")[0]


def read_csv(csv_path):
    with open(csv_path, 'r') as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        dic = {str(row[0]): str(row[1]) for _, row in enumerate(reader)}
    return dic


def create_tmp_dir(path):
    millis = int(round(time.time() * 1000))
    tid = threading.current_thread()
    tmp_dir_path = os.path.join(path, str(tid), str(millis))
    os.makedirs(tmp_dir_path)
    return tmp_dir_path


def is_empty_dir(path):
    return len(dir_to_file_list(path)) == 0


def write_json(file, json_obj):
    with open(file, 'w') as f:
        json.dump(json_obj, f, indent=4, separators=(',', ': '))


def read_json(file):
    return json.load(open(file))


if __name__ == '__main__':
    print(dir_to_file_list_with_ext('D:/private/datasets/handTrack/studies/full_hands/17.12.17/20', "avi"))