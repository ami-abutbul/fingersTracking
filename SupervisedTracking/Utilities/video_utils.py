import cv2
import os.path
from Utilities.file_utils import create_dir


def extract_frames_from_video(video_path, amount_of_frames=None, start_index_name=0, warm_frames=None, fps=None):
    out_dir = os.path.abspath(os.path.join(os.path.join(video_path, os.pardir), "frames"))
    create_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if amount_of_frames is not None:
        taken_frames = total_frames // amount_of_frames
        if taken_frames == 0:
            taken_frames = 1
        remainder = total_frames / amount_of_frames - taken_frames
    elif fps is not None:
        amount = (total_frames * fps) / cap.get(cv2.CAP_PROP_FPS)
        taken_frames = total_frames // amount
        if taken_frames == 0:
            taken_frames = 1
        remainder = total_frames / amount - taken_frames
    else:
        taken_frames = 1
        remainder = 0

    if warm_frames is not None:
        warm_frames -= 1

    success, image = cap.read()
    index = start_index_name
    count = 0
    remainder_count = 0
    success = True
    while success:
        if count == 0:
            remainder_count += remainder
            if remainder_count >= 1:
                count = taken_frames + 1
                remainder_count -= 1
            else:
                count = taken_frames

            cv2.imwrite(os.path.join(out_dir, "{}.jpg".format(index)), image)

        count -= 1
        success, image = cap.read()
        index += 1
        if warm_frames is not None and warm_frames > 0:
            count = 0
            remainder_count = 0
            warm_frames -= 1


def number_of_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


if __name__ == '__main__':
    extract_frames_from_video("D:/bitbucket/projectHandTrack/BoardGame/workFolder/0/video.avi", 35)