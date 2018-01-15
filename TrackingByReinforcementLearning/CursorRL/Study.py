from Utilities.two_dimension_utils import p2p_distance, in_circle, get_point_in_distance
from CursorRL.configuration import *
from Utilities.file_utils import dir_to_file_list, get_file_name, read_json, get_file_name_without_ext
from Utilities.image_utils import image_to_mat
from collections import namedtuple
from PIL import Image
import numpy as np
import json
import re


Point = namedtuple('Point', ['x', 'y'])
Action = namedtuple('Action', ['direction', 'step_size', 'click'])


class Status(object):
    PLACE_GOOD = 0
    PLACE_OK = 1
    PLACE_BAD = 2
    CLICK_GOOD = 3
    CLICK_BAD = 4
    CLICK_MISSING = 5
    WAITING = 6


class Landmark(object):
    def __init__(self, point_tuple, click, stay):
        self.point = Point(point_tuple[0], point_tuple[1])
        self.click = click
        self.stay = stay

    def distance(self, landmark=None, point=None):
        res = None
        if landmark is not None:
            assert (isinstance(landmark, Landmark))
            res = p2p_distance(self.point.x, self.point.y, landmark.point.x, landmark.point.y)

        elif point is not None:
            assert (isinstance(point, Point))
            res = p2p_distance(self.point.x, self.point.y, point.x, point.y)

        return res

    def get_status(self, point):
        assert (isinstance(point, Point))
        if in_circle(self.point.x, self.point.y, good_radius, point.x, point.y):
            return Status.PLACE_GOOD

        if in_circle(self.point.x, self.point.y, ok_radius, point.x, point.y):
            return Status.PLACE_OK

        return Status.PLACE_BAD


class Path(object):
    def __init__(self, path_as_json):
        self.screen_resolution, self.start_point_idx, self.fps, self.landmarks = Path._load_path_from_json(path_as_json)
        self.current_point = self.landmarks[self.start_point_idx].point
        self.current_landmark_index = 0
        self.path_len = len(self.landmarks)

    def is_end(self):
        return self.current_landmark_index >= self.path_len

    def update_closest_landmark(self, validate_indexes=True):
        vec_distance = np.vectorize(lambda landmark: landmark.distance(point=self.current_point))
        distances = vec_distance(self.landmarks)
        if validate_indexes:
            distances = distances[self.current_landmark_index:]
            self.current_landmark_index = np.array(distances).argmin() + self.current_landmark_index
        else:
            self.current_landmark_index = np.array(distances).argmin()

    def next_landmark(self):
        self.current_landmark_index += 1

    def get_status(self):
        landmark = self.landmarks[self.current_landmark_index]
        return landmark.get_status(self.current_point)

    def is_click_point(self):
        if self.current_landmark_index >= self.path_len:
            return False
        return self.landmarks[self.current_landmark_index].click

    def is_stay_point(self):
        if self.current_landmark_index >= self.path_len:
            return False
        return self.landmarks[self.current_landmark_index].stay

    def set_current_point(self, point):
        x = point.x if point.x >= 0 else 0
        y = point.y if point.y >= 0 else 0
        x = x if x <= self.screen_resolution[0] else self.screen_resolution[0]
        y = y if y <= self.screen_resolution[1] else self.screen_resolution[1]
        self.current_point = Point(x, y)

    def get_GT_step_size(self):
        curr_index = self.current_landmark_index
        if curr_index + 1 >= self.path_len:
            return 0
        return self.landmarks[curr_index].distance(landmark=self.landmarks[curr_index + 1])

    @classmethod
    def _load_path_from_json(cls, json_file):
        path_json = read_json(json_file)
        screen_resolution = path_json['screen_resolution']
        start_point_index = path_json['start_point_index']
        frames_per_sec = path_json['frames_per_sec']
        landmarks = list(map(lambda lm_json: Landmark(lm_json['point'], lm_json['click'], lm_json['stay']),
                                             path_json['landmarks']))
        return screen_resolution, start_point_index, frames_per_sec, landmarks


class Study(object):
    def __init__(self, study_dir):
        self.study_dir = study_dir
        self.frames, self.path = Study._load_study_from_dir(study_dir)
        self.frames_len = len(self.frames)
        self.image_index = warm_frames_num
        self.clickable_flag = self.path.is_click_point()
        self.double_click = self.path.is_click_point()
        self.clickable_counter = int(self.path.fps * sec_waiting_for_click) if self.clickable_flag else None
        self.finished_successfully = False
        self.actions = []
        self.rewards = []

    def step(self, action):
        if action.click is True:
            if self.clickable_flag:
                if self.double_click:
                    self.double_click = False
                    self.clickable_counter = int(self.path.fps * sec_waiting_for_click)
                    next_image = self.get_image()
                    done = True if next_image is None else False
                    return next_image, Study._reward(Status.CLICK_GOOD), done
                else:
                    self.clickable_flag = False
                    self.clickable_counter = None
                    self.finished_successfully = True
                    return None, Study._reward(Status.CLICK_GOOD), True
            else:
                return None, Study._reward(Status.CLICK_BAD), True

        if action.direction is not None and action.step_size is not None:
            if self.clickable_flag:
                return None, Study._reward(Status.CLICK_BAD), True

            if self.path.is_stay_point():
                return None, Study._reward(Status.PLACE_BAD), True

            x0 = self.path.current_point.x
            y0 = self.path.current_point.y
            x1, y1 = get_point_in_distance(x0, y0, Point(action.direction[0], action.direction[1]), action.step_size)
            self.path.set_current_point(Point(x1, y1))
            self.path.next_landmark()
            self.clickable_flag = self.path.is_click_point()
            if self.clickable_flag:
                self.clickable_counter = int(self.path.fps * sec_waiting_for_click)
            status = self.path.get_status()
            done = (self.path.is_end() and not self.clickable_flag) or self.is_last_image() or status == Status.PLACE_BAD
            next_image = None if done else self.get_image()
            self.finished_successfully = self.is_last_image() and status != Status.PLACE_BAD
            return next_image, Study._reward(status), done

        if action.direction is None and action.step_size is None and action.click is None:
            if self.clickable_flag:
                self.clickable_counter -= 1
                if self.clickable_counter == 0:
                    return None, Study._reward(Status.CLICK_MISSING), True
                next_image = self.get_image()
                done = True if next_image is None else False
                return next_image, Study._reward(Status.WAITING), done

            if not self.path.is_stay_point():
                return None, Study._reward(Status.PLACE_BAD), True

            done = self.is_last_image()
            next_image = None if done else self.get_image()
            self.finished_successfully = done
            return next_image, Study._reward(Status.WAITING), done

    def is_last_image(self):
        return self.image_index >= self.frames_len

    def get_image_by_index(self, index, expand_dim=False):
        if index >= self.frames_len:
            return None
        mat_image = image_to_mat(Image.open(self.frames[index]), shape=(image_height, image_width, 3))
        im = Study._image_pre_processing(mat_image)
        if expand_dim:
            im = np.expand_dims(im, axis=0)
        return im

    def get_image(self):
        mat_image = self.get_image_by_index(self.image_index)
        self.image_index += 1
        if mat_image is None:
            return None
        else:
            return np.expand_dims(mat_image, axis=0)

    def get_warm_frames(self):
        return [self.get_image_by_index(i) for i in range(warm_frames_num)]

    def get_GT_step_size(self):
        return self.path.get_GT_step_size()

    def succeeded_steps(self):
        return self.image_index - warm_frames_num

    @classmethod
    def _reward(cls, status):
        return rewards[status]

    @classmethod
    def _load_study_from_dir(cls, study_dir):
        file_list = dir_to_file_list(os.path.join(study_dir, "frames"))
        path_file = os.path.join(study_dir, "Path.json")
        path = Path(path_file)
        file_list = [(int(get_file_name_without_ext(x)), x) for x in file_list]
        file_list = sorted(file_list, key=lambda x: x[0])
        file_list = [x[1] for x in file_list]
        return file_list, path

    @classmethod
    def _image_pre_processing(cls, image):
        return image / 255.

    def __lt__(self, other):
        return self.image_index < other.image_index

    # def _image_index_after_click(self):
    #     current_index = self.image_index
    #     while "frame_1000" not in self.frames[current_index]:
    #         current_index += 1
    #     return current_index
