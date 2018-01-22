from Utilities.two_dimension_utils import p2p_distance, in_circle, get_point_in_distance
from Utilities.file_utils import dir_to_file_list, dir_to_subdir_list, read_json, get_file_name_without_ext
from Utilities.image_utils import image_to_mat
from Model.configuration import *
from collections import namedtuple
from random import shuffle
from PIL import Image
import numpy as np
import os

Point = namedtuple('Point', ['x', 'y'])


class Landmark(object):
    def __init__(self, x, y, click=None):
        self.point = Point(x, y)
        self.click = click

    def distance(self, landmark=None, point=None):
        res = None
        if landmark is not None:
            assert (isinstance(landmark, Landmark))
            res = p2p_distance(self.point.x, self.point.y, landmark.point.x, landmark.point.y)

        elif point is not None:
            assert (isinstance(point, Point))
            res = p2p_distance(self.point.x, self.point.y, point.x, point.y)

        return res

    def point(self):
        return self.point

    def relative_vector_distance(self, point):
        return Point(point.x - self.point.x, point.y - self.point.y)


class Path(object):
    def __init__(self, path_file):
        self.landmarks = Path._load_path(path_file)
        self.current_point = self.landmarks[0]
        self.current_landmark_index = 0
        self.path_len = len(self.landmarks)

    def is_end(self):
        return self.current_landmark_index >= self.path_len

    def is_click_point(self):
        if self.current_landmark_index >= self.path_len:
            return False
        return self.landmarks[self.current_landmark_index].click

    def get_relative_step(self):
        curr_index = self.current_landmark_index
        self.current_landmark_index += 1
        if curr_index + 1 >= self.path_len:
            return Point(0, 0)
        return self.landmarks[curr_index].relative_vector_distance(self.landmarks[curr_index + 1].point)

    @classmethod
    def _load_path(cls, file):
        path = []
        with open(file, "r") as file:
            for line in file:
                line = line.split(",")
                path.append((int(line[0]), int(line[1].rstrip('\n'))))
        landmarks = list(map(lambda point: Landmark(point[0], point[1]), path))
        return landmarks


class Study(object):
    def __init__(self, study_dir):
        self.study_dir = study_dir
        self.frames, self.path = Study._load_study_from_dir(study_dir)
        self.frames_len = len(self.frames)
        self.image_index = warm_frames_num + 1  # shift one image
        self.finished_successfully = False

    def is_last_image(self):
        return self.image_index >= self.frames_len

    def is_end_of_study(self):
        return self.is_last_image() or self.path.is_end()

    def get_image_by_index(self, index, expand_dim=False):
        if index >= self.frames_len:
            return None
        mat_image = image_to_mat(Image.open(self.frames[index]), shape=(image_height, image_width, 3))
        im = Study._image_pre_processing(mat_image)
        if expand_dim:
            im = np.expand_dims(im, axis=0)
        return im

    def get_image(self, advance_counter=True):
        mat_image = self.get_image_by_index(self.image_index)
        if advance_counter:
            self.image_index += 1
        if mat_image is None:
            return None
        else:
            return np.expand_dims(mat_image, axis=0)

    def get_relative_step(self):
        return self.path.get_relative_step()

    def next(self):
        return self.get_image(), self.get_relative_step()

    def next_couple(self):
        frame1 = self.get_image()
        direction_vec = self.get_relative_step()
        frame2 = self.get_image(advance_counter=False)
        return frame1, frame2, direction_vec

    def get_warm_frames(self):
        return [self.get_image_by_index(i) for i in range(warm_frames_num)]

    @classmethod
    def _load_study_from_dir(cls, study_dir):
        file_list = dir_to_file_list(os.path.join(study_dir, "frames"))
        path_file = os.path.join(study_dir, "Path.txt")
        path = Path(path_file)
        file_list = [(int(get_file_name_without_ext(x)), x) for x in file_list]
        file_list = sorted(file_list, key=lambda x: x[0])
        file_list = [x[1] for x in file_list]
        return file_list, path

    @classmethod
    def _image_pre_processing(cls, image):
        im = image / 255.
        return im - np.mean(im)
        # return image / 255.


class StudiesHandler(object):
    def __init__(self, studies_dir, list_of_dirs=False):
        if list_of_dirs:
            studies = []
            for dir in studies_dir:
                studies += dir_to_subdir_list(dir)
            self.studies = studies
        else:
            self.studies = dir_to_subdir_list(studies_dir)
        self.current_study_index = 0
        self.current_study = None
        self.epoch_done = False

    def get_study(self):
        self.current_study = Study(self.studies[self.current_study_index])
        self.current_study_index += 1
        if self.current_study_index == len(self.studies):
            self.current_study_index = 0
            shuffle(self.studies)
            self.epoch_done = True
        return self.current_study

    @classmethod
    def write_list_to_file(cls, data_list, file_path):
        with open(file_path, "a") as file:
            for item in data_list:
                file.write(str(item) + "\n")
