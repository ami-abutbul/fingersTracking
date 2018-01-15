import numpy as np
from CursorRL.configuration import *
import heapq as heap


class ExperiencedStudiesBuffer(object):
    def __init__(self, size):
        self.size = size
        self.studies = {} #[]
        self.current_study = None
        self.image_index = None
        self.trace_index = None
        self.full_traces_amount = None

    def append(self, study):
        if study.study_dir in self.studies:
            new_study_reward = np.sum(study.rewards)
            old_study_reward = np.sum(self.studies[study.study_dir].rewards)
            if new_study_reward > old_study_reward:
                self.studies[study.study_dir] = study
            elif new_study_reward == old_study_reward and study.succeeded_steps() > self.studies[study.study_dir].succeeded_steps():
                self.studies[study.study_dir] = study
        else:
            self.studies[study.study_dir] = study

        # lru
        # if len(self.studies) >= self.size:
        #     self.studies.pop(0)
        # self.studies.append(study)

        # mean heap
        #     heap.heappop(self.studies)
        # heap.heappush(self.studies, study)

    def select_study(self):
        # self.current_study = np.random.choice(self.studies)
        self.current_study = self.studies[np.random.choice(list(self.studies.keys()))]
        self.trace_index = 0
        self.image_index = warm_frames_num
        self.full_traces_amount = (self.current_study.image_index - warm_frames_num) // max_trace_len

    def get_warm_frames(self):
        return self.current_study.get_warm_frames()

    def get_trace(self):
        trace = []
        done = False
        if self.trace_index == self.full_traces_amount:
            trace_length = (self.current_study.image_index - warm_frames_num) % max_trace_len
            # trace_length -= 1
            done = True
        else:
            trace_length = max_trace_len
            # self.trace_index += max_trace_len

        for i in range(trace_length):
            index = i + self.image_index
            next_image = self.current_study.get_image_by_index(index + 1, expand_dim=True)
            if next_image is None:
                trace_length -= 1
                done = True
                break
            if index - warm_frames_num >= len(self.current_study.actions) and index - warm_frames_num >= len(self.current_study.rewards):
                trace_length -= 1
                done = True
                break
            trace.append([self.current_study.get_image_by_index(index, expand_dim=True),
                          self.current_study.actions[index - warm_frames_num],
                          self.current_study.rewards[index - warm_frames_num],
                          next_image,
                          False])
        self.trace_index += 1
        self.image_index += trace_length
        if done:
            if len(trace) == 0:
                return None, None, None
            trace[-1][-1] = True
        trace = np.array(trace)
        return trace, len(trace), done

