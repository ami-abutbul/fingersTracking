from Utilities.file_utils import dir_to_subdir_list, create_dir, delete_dir
from CursorRL.Study import Study, Action
from CursorRL.configuration import *
from CursorRL.actions import *
from random import shuffle
import os


class Environment(object):
    def __init__(self, studies_dir, stats_dir, list_of_dirs=False):
        self.WRITE_STATS_STEPS = 100
        if list_of_dirs:
            studies = []
            for dir in studies_dir:
                studies += dir_to_subdir_list(dir)
            self.studies = studies
        else:
            self.studies = dir_to_subdir_list(studies_dir)
        self.current_study_index = 0
        self.current_study = None
        self.rewards_counter = 0
        self.completed_paths_counter = 0

        delete_dir(stats_dir)
        create_dir(stats_dir)
        self.steps_counter = 0
        self.rewards = []
        self.path_len = []
        self.path_completed = []
        self.rewards_file = os.path.join(stats_dir, "rewards.txt")
        self.path_len_file = os.path.join(stats_dir, "path_len.txt")
        self.path_completed_file = os.path.join(stats_dir, "path_completed.txt")

    def reset(self):
        if self.steps_counter == self.WRITE_STATS_STEPS:
            self.steps_counter = 0
            Environment.write_list_to_file(self.rewards, self.rewards_file)
            Environment.write_list_to_file(self.path_len, self.path_len_file)
            Environment.write_list_to_file(self.path_completed, self.path_completed_file)
            self.rewards = []
            self.path_len = []
            self.path_completed = []

        if self.current_study is not None:
            self.rewards.append(self.rewards_counter)
            self.path_len.append(self.current_study.image_index - warm_frames_num)
            self.path_completed.append(self.completed_paths_counter)

        self.current_study = Study(self.studies[self.current_study_index])
        self.current_study_index += 1
        if self.current_study_index == len(self.studies):
            self.current_study_index = 0
            shuffle(self.studies)
        self.steps_counter += 1
        return self.current_study.get_warm_frames()

    def step(self, action_index, step_size=None):
        self.current_study.actions.append(action_index)
        if action_index == none_action:
            s, r, d = self.current_study.step(Action(None, None, None))
        elif action_index == click_action:
            s, r, d = self.current_study.step(Action(None, None, True))
        else:
            if step_size is None:
                step_size = self.current_study.get_GT_step_size()
            s, r, d = self.current_study.step(Action(directions[action_index - 2], step_size, None))
        self.current_study.rewards.append(r)
        self.rewards_counter += r
        if d:
            if self.current_study.finished_successfully:
                self.completed_paths_counter += 1

        return s, r, d

    def start(self):
        if self.current_study is not None:
            return self.current_study.get_image()

    def write_stats(self):
        Environment.write_list_to_file(self.rewards, self.rewards_file)
        Environment.write_list_to_file(self.path_len, self.path_len_file)
        Environment.write_list_to_file(self.path_completed, self.path_completed_file)

    @classmethod
    def write_list_to_file(cls, data_list, file_path):
        with open(file_path, "a") as file:
            for item in data_list:
                file.write(str(item) + "\n")
