import pygame
import numpy as np
import os
from BoardGame.FunctionGenerator import FunctionGenerator
from BoardGame.configuration import *
from win32api import GetSystemMetrics
from Utilities.file_utils import write_json, create_dir, dir_to_file_list_with_ext, read_json, get_file_name, dir_to_subdir_list
from Utilities.video_utils import extract_frames_from_video
import cv2

RESOLUTION = [GetSystemMetrics(0), GetSystemMetrics(1)]  # [Width, Height]


class Board(object):
    PATH_NAME = "Path.json"
    FILM_FPS = 20.0

    # Define the basic colors in RGB format
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    def __init__(self, work_dir, fingers_functions=False):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME | pygame.FULLSCREEN)
        self.screen.fill(Board.WHITE)
        self.functionGen = FunctionGenerator(fingers_functions)
        create_dir(work_dir)
        self.work_folder = work_dir
        self.cap = cv2.VideoCapture(0)
        self.frameWriter = None

    def _draw_function(self, x_points, y_points, file):
        path_json = dict()
        path_json['screen_resolution'] = RESOLUTION
        path_json['frames_per_sec'] = fps
        path_json['click_point_index'] = -1
        path_json['start_point_index'] = 0
        start_point_index = np.random.choice([0, len(x_points) - 1])
        stay_point_index = None
        click_point_index = None
        if len(x_points) == 1:
            if np.random.rand(1) < 0.5:
                stay_point_index = 0
            else:
                click_point_index = 0
        elif np.random.rand(1) < 0.5:
            click_point_index = len(x_points) - 1  # click point always be at the end

        prev_x = x_points[0]
        prev_y = y_points[0]
        landmarks = []
        for xp, yp in zip(x_points, y_points):
            point = {'point': [np.asscalar(xp), np.asscalar(yp)], 'click': False, 'stay': False}
            landmarks.append(point)
            pygame.draw.circle(self.screen, Board.BLUE, [xp, yp], 5)
            pygame.draw.line(self.screen, Board.BLUE, [prev_x, prev_y], [xp, yp], 3)
            prev_x = xp
            prev_y = yp

        if start_point_index != 0:
            landmarks = list(reversed(landmarks))
        if stay_point_index is not None:
            landmarks[stay_point_index]["stay"] = True
        if click_point_index is not None:
            pygame.draw.circle(self.screen, Board.GREEN, landmarks[click_point_index]['point'], 8)
            landmarks[click_point_index]["click"] = True
            path_json['click_point_index'] = click_point_index
        if path_json['click_point_index'] != 0:
            pygame.draw.circle(self.screen, Board.RED, landmarks[0]['point'], 8)

        path_json['landmarks'] = landmarks
        write_json(file, path_json)

    def play(self, index):
        i = index
        done = False
        while not done:
            ret, frame = self.cap.read()
            if self.frameWriter is not None and ret:
                frame = cv2.flip(frame, 1)
                self.frameWriter.write(frame)

            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LCTRL:
                        self.screen.fill(Board.WHITE)
                        x_points, y_points = self.functionGen.generate_function()

                        i += 1
                        video_index = 1
                        base_dir = os.path.join(self.work_folder, str(i))
                        create_dir(base_dir)
                        create_dir(os.path.join(base_dir, "frames"))

                        self._draw_function(x_points, y_points, os.path.join(base_dir, Board.PATH_NAME))
                        pygame.image.save(self.screen, os.path.join(base_dir, "screenshot.jpg"))

                        if self.frameWriter is not None:
                            self.frameWriter.release()
                            self.frameWriter = None

                        if self.frameWriter is None:
                            pygame.draw.rect(self.screen, Board.RED, (0, 0, 20, 20))
                        else:
                            pygame.draw.rect(self.screen, Board.GREEN, (0, 0, 20, 20))

                    if event.key == pygame.K_LSHIFT:
                        self.frameWriter = cv2.VideoWriter(os.path.join(self.work_folder, str(i), "video_{}.avi".format(video_index)),
                                                           cv2.VideoWriter_fourcc(*"XVID"), Board.FILM_FPS, (640, 480))
                        pygame.draw.rect(self.screen, Board.GREEN, (0, 0, 20, 20))

                    if event.key == pygame.K_z:
                        if self.frameWriter is not None:
                            self.frameWriter.release()
                        video_index += 1
                        self.frameWriter = cv2.VideoWriter(os.path.join(self.work_folder, str(i), "video_{}.avi".format(video_index)),
                                                           cv2.VideoWriter_fourcc(*"XVID"), Board.FILM_FPS, (640, 480))
                        pygame.draw.rect(self.screen, Board.GREEN, (0, 0, 20, 20))

                if event.type == pygame.MOUSEBUTTONUP or event.type == pygame.K_ESCAPE:
                    done = True
                    self.cap.release()
                    cv2.destroyAllWindows()
                    if self.frameWriter is not None:
                        self.frameWriter.release()
                        self.frameWriter = None
                        pygame.draw.rect(self.screen, Board.RED, (0, 0, 20, 20))

            pygame.display.flip()

    @classmethod
    def extract_frames(cls, study_dir):
        path = read_json(os.path.join(study_dir, Board.PATH_NAME))
        points_amount = len(path['landmarks'])
        videos_list = dir_to_file_list_with_ext(study_dir, "avi")

        if points_amount == 1:
            if len(videos_list) != 1:
                print("Error: " + get_file_name(study_dir))
                return
            video_path = os.path.join(study_dir, "video_1.avi")
            extract_frames_from_video(video_path, fps=fps)
            return

        if path['click_point_index'] == -1:
            if len(videos_list) != 1:
                print("Error: " + get_file_name(study_dir))
                return
            video_path = os.path.join(study_dir, "video_1.avi")
            extract_frames_from_video(video_path, points_amount, warm_frames=warm_frames_num)
        else:
            if len(videos_list) != 2:
                print("Error: " + get_file_name(study_dir))
                return
            video_path = os.path.join(study_dir, "video_1.avi")
            extract_frames_from_video(video_path, path['click_point_index'], 0, warm_frames=warm_frames_num)
            video_path = os.path.join(study_dir, "video_2.avi")
            extract_frames_from_video(video_path, fps, 1000)

if __name__ == '__main__':
    # work_folder = "D:/private/datasets/handTrack/dummy"
    # Board(work_folder, fingers_functions=False).play(index=0)

    # extract each video to frames
    print("extracting frames...")
    sub_dirs = dir_to_subdir_list("D:/private/datasets/handTrack/dummy")
    for dir in sub_dirs:
        # if not os.listdir(os.path.join(dir, "frames")):  # if  empty
        #     print(dir)
        # Board.extract_frames(dir)

        video_path = os.path.join(dir, "video_1.avi")
        extract_frames_from_video(video_path)



    # sub_dirs = dir_to_subdir_list("D:/private/datasets/handTrack/studies/fingers/20.12.17")
    # for dir in sub_dirs:
    #     files = os.listdir(os.path.join(dir, "frames"))
    #     if len(files) <= 5:
    #         print(dir)
