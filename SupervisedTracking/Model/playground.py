__author__ = "Ami Abutbul"
from Model.Study import StudiesHandler
from Utilities.image_utils import mat_to_image

if __name__ == '__main__':
    studiesHandler = StudiesHandler("C:/Users/il115552/Desktop/New folder (6)")
    study = studiesHandler.get_study()
    print("study.is_end_of_study: {}".format(study.is_end_of_study()))
    while not study.is_end_of_study():
        im, point = study.next()
        im = im * 255.
        mat_to_image(im, shape=(480, 640, 3)).show()