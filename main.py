import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from table_feature_extractor import get_table_key_point
from box_feature_extractor import get_box_key_point
from solver import get_answer

def process(path_to_image, show):
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform1, transform2, transform3 = get_table_key_point(img, show)
    h, w, flag =  get_box_key_point(img, transform1, transform2, transform3, show)
    if h is None or w is None:
        return None
    answer = get_answer(transform1, transform2, transform3, h, w, flag, img, show)
    return answer

if __name__ == '__main__':
    folder = './dataset/Yes/'
    img_names = os.listdir(folder)
    answers = []
    for name in img_names:
        path_to_image = folder + name
        ans = process(path_to_image, False)
        answers.append(ans)
        print(name, ans)
    print(answers)