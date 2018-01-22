import pygame
import cv2
import time
import os
import numpy as np
from Utilities.file_utils import dir_to_subdir_list, create_dir, delete_dir
from Utilities.video_utils import extract_frames_from_video


class Board(object):
    FILM_FPS = 20.0
    warm_frames_amount = 4
    REPEAT = 5

    # Define the basic colors in RGB format
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BACKGROUND = (196, 196, 196)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME | pygame.FULLSCREEN)
        self.screen.fill(Board.BACKGROUND)
        self.cap = cv2.VideoCapture(0)
        self.frameWriter = None

    def record_path(self, work_folder, path_index):
        done = False
        curr_t = 0.0
        start = False
        path = []

        while not done:
            if start:
                if time.time() * 1000 - curr_t >= 1000 / Board.FILM_FPS:
                    curr_t = time.time() * 1000
                    pos = pygame.mouse.get_pos()
                    pygame.draw.circle(self.screen, Board.RED, pos, 3)
                    path.append(pos)

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    start = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LCTRL:
                        start = False
                        base_dir = os.path.join(work_folder, str(path_index))
                        create_dir(base_dir)
                        Board.write_path_to_file(os.path.join(base_dir, "Path.txt"), path)
                        pygame.image.save(self.screen, os.path.join(base_dir, "screenshot.jpg"))
                        self.screen.fill(Board.BACKGROUND)
                        path_index += 1
                        path = []

                    if event.key == pygame.K_ESCAPE:
                        done = True

            pygame.display.flip()

    def play(self, path_folder, work_folder):
        done = start = False
        t = 0.0
        path_files = dir_to_subdir_list(path_folder)
        path = []
        path_index = 0
        study_index = 132
        # <-
        repeat_index = Board.REPEAT
        while not done:
            if start:
                if time.time() * 1000 - t >= 1000 / Board.FILM_FPS:
                    t = time.time() * 1000
                    _, frame = self.cap.read()
                    self.frameWriter.write(frame)
                    self.screen.fill(Board.BACKGROUND)
                    pygame.draw.circle(self.screen, Board.RED, [*path[path_index]], 8)
                    path_index = path_index + 1

                    if path_index == len(path):
                        pygame.draw.rect(self.screen, Board.RED, (800, 0, 20, 20))
                        start = False

            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LSHIFT:
                        study_index += 1
                        base_dir = os.path.join(work_folder, str(study_index))
                        create_dir(base_dir)
                        Board.write_path_to_file(os.path.join(base_dir, "Path.txt"), path)
                        if self.frameWriter is not None:
                            self.frameWriter.release()
                        self.frameWriter = cv2.VideoWriter(os.path.join(work_folder, str(study_index), "video.avi"),
                                                           cv2.VideoWriter_fourcc(*"XVID"), Board.FILM_FPS, (640, 480))

                        self.cap.read()  # clear old frame
                        for i in range(Board.warm_frames_amount):
                            _, frame = self.cap.read()
                            self.frameWriter.write(frame)
                        start = True

                    if event.key == pygame.K_LCTRL:
                        self.screen.fill(Board.BACKGROUND)
                        path_index = 0
                        start = False
                        if repeat_index == Board.REPEAT:
                            repeat_index = 0
                            current_file = np.random.choice(path_files)
                        repeat_index += 1
                        path = Board.read_path_from_file(os.path.join(current_file, "Path.txt"))

                        pygame.draw.circle(self.screen, Board.RED, [*path[0]], 8)
                        for i in range(len(path) - 1):
                            pygame.draw.line(self.screen, Board.BLUE, [*path[i]], [*path[i + 1]], 3)

                    if event.key == pygame.K_d:
                        self.screen.fill(Board.BACKGROUND)
                        path_index = 0
                        start = False
                        create_dir(os.path.join(work_folder, str(study_index), "DELETE"))
                        path = Board.read_path_from_file(os.path.join(current_file, "Path.txt"))

                        pygame.draw.circle(self.screen, Board.RED, [*path[0]], 8)
                        for i in range(len(path) - 1):
                            pygame.draw.line(self.screen, Board.BLUE, [*path[i]], [*path[i + 1]], 3)

                    if event.key == pygame.K_ESCAPE:
                        done = True
                        self.cap.release()
                        cv2.destroyAllWindows()
            pygame.display.flip()

    @classmethod
    def write_path_to_file(cls, file, path):
        with open(file, "w") as file:
            for item in path:
                file.write("{},{}\n".format(item[0], item[1]))

    @classmethod
    def read_path_from_file(cls, path_file):
        path = []
        x_bias = np.random.randint(0, 100) * np.random.choice([-1, 1])
        y_bias = np.random.randint(0, 50) * np.random.choice([-1, 1])
        with open(path_file, "r") as file:
            for line in file:
                line = line.split(",")
                path.append((int(line[0]) + x_bias, int(line[1].rstrip('\n')) + y_bias))
        return np.array(path)

if __name__ == '__main__':
    # Board().record_path("D:/private/datasets/movingPoint/paths")
    # Board().play("D:/private/datasets/movingPoint/paths", "D:/private/datasets/movingPoint/studies")

    sub_dirs = dir_to_subdir_list("D:/private/datasets/movingPoint/studies")
    for dir in sub_dirs:
        video_path = os.path.join(dir, "video.avi")
        extract_frames_from_video(video_path)

    # sub_dirs = dir_to_subdir_list("D:/private/datasets/movingPoint/studies9.1.18")
    # for dir in sub_dirs:
    #     dir_list = dir_to_subdir_list(dir)
    #     if len(dir_list) > 0 and "DELETE" in dir_list[0]:
    #         delete_dir(dir)



