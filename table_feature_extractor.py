import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt


def template_creator(path_to_templates = 'templates.pkl'):
    # подготовка темплейто
    with open(path_to_templates, 'rb') as fp:
        data = pickle.load(fp)

    for elem in data:
        pt, size, angle, response, octave, class_id = data[elem]['kp']
        collect_kp = [cv2.KeyPoint(x=pt[i][0],y=pt[i][1],_size=size[i], _angle=angle[i],
                                _response=response[i], _octave=octave[i], _class_id=class_id[i])
                     for i in range(len(pt))]
        data[elem]['kp'] = collect_kp
    return data

def get_best_template(des, data):
    best_template = ''
    min_distance = 10000

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for elem in data:
        des_template = data[elem]['des']
        # Match descriptors.
        matches = bf.match(des_template, des)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        distances = [elem.distance for elem in matches]
        avg_dist = sum(distances) / len(distances)
        if avg_dist < min_distance:
            min_distance = avg_dist
            best_template = elem
    return best_template


def find_best_points(kp,des, data, best_template):
    point_template = data[best_template]['points']
    key_point_template = data[best_template]['kp']
    des_template = data[best_template]['des']
    # Match descriptors.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    src = np.float32([ key_point_template[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    points1 = np.float32(point_template[0]).reshape(-1, 1, 2)
    points2 = np.float32(point_template[1]).reshape(-1, 1, 2)
    points3 = np.float32(point_template[2]).reshape(-1, 1, 2)
    # гомография и перспективные искажения
    M, status = cv2.findHomography(src, dst, cv2.RANSAC, 2)
    transform1 = cv2.perspectiveTransform(points1, M)
    transform2 = cv2.perspectiveTransform(points2, M)
    transform3 = cv2.perspectiveTransform(points3, M)
    return transform1, transform2, transform3


def get_table_key_point(img, show=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # находим особые точки и их дескрипторы
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img_gray, None)
    data = template_creator()
    best_template = get_best_template(des, data)
    transform1, transform2, transform3 = find_best_points(kp, des, data, best_template)

    if show:
        # изобразим полученное
        # найденные особые точки
        a = transform1[0][0]
        b = transform1[1][0]
        c = transform1[2][0]
        d = transform1[3][0]

        a1 = transform2[0][0]
        b1 = transform2[1][0]
        c1 = transform2[2][0]
        d1 = transform2[3][0]

        e = transform3[1][0]
        g = transform3[2][0]
        plt.subplots(figsize=(7, 7))

        cv2.polylines(img_gray, [np.int32(transform1)], True, 0, 5, cv2.LINE_AA)
        cv2.polylines(img_gray, [np.int32(transform2)], True, 0, 5, cv2.LINE_AA)
        cv2.polylines(img_gray, [np.int32(transform3)], True, 0, 5, cv2.LINE_AA)
        plt.title('Изображение')
        plt.imshow(img_gray, cmap='gray')
        plt.plot([a[0], b[0], c[0], d[0], a1[0], b1[0], c1[0], d1[0], e[0], g[0]],
                 [a[1], b[1], c[1], d[1], a1[1], b1[1], c1[1], d1[1], e[1], g[1]], 'ob', markersize=5)
        plt.show()
    return transform1, transform2, transform3