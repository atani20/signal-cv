from skimage.morphology import binary_closing
import random
import cv2
import numpy as np
from math import cos, tan
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from sklearn.cluster import KMeans


def get_edges(img, mask, show):
    img_edges = canny(rgb2gray(img), sigma=1.4, low_threshold=0.00001, mask=mask)
    if show:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Source")
        ax[1].set_title("Edges")
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img_edges, cmap="gray")
        plt.show()
    return img_edges

# расширение маски
def expand_mask(mask, rad=13, show=False):
    new_mask = np.zeros(mask.shape, dtype=bool)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[0]):
            if mask[i, j]:
                for i1 in range(max(i - rad // 2, 0), min(i + rad // 2, mask.shape[0])):
                    for j1 in range(max(0, j - rad // 2), min(j + rad // 2, mask.shape[0])):
                        new_mask[i1, j1] = True
    if show:
        plt.imshow(new_mask, cmap='gray')
        plt.show()
    return new_mask

def get_box_mask(img,transform1, transform2, transform3, show):
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    mask = (r > g) * (g - b < 20) * (g - b > -5) * (r > 70)  * (g < 90)  # было подобрано исходя из цветового анализа изображения
    mask = np.uint8(mask)*255
    cv2.fillPoly(mask, [np.int32(transform1)], 0)
    cv2.fillPoly(mask, [np.int32(transform2)], 0)
    cv2.fillPoly(mask, [np.int32(transform3)], 0)

    labels = label(mask) # разбиение маски на компоненты связности
    props = regionprops(labels) # нахождение свойств каждой области (положение центра, площадь, bbox, интервал интенсивностей и т.д.)
    areas = [prop.area for prop in props] # нас интересуют площади компонент связности
    areas = np.array(areas)
    largest_comp_id = np.array(areas).argmax()# находим номер компоненты с максимальной площадью
    mask = labels == (largest_comp_id + 1)
    mask = binary_closing(mask, selem=np.ones((15, 15)))
    mask = convex_hull_image(mask)
    mask = expand_mask(mask, 5)
    if show:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Source")
        ax[1].set_title("Box region")
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(mask, cmap="gray")
        plt.show()
    return mask

def show_box_point(image, h, w):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    ax[0].set_title('Image')
    ax[0].imshow(image, cmap='gray')

    if w is not None:
        ax[1].set_title('main points width')
        ax[1].imshow(image, cmap='gray')
        ax[1].plot(w[:, 1], w[:, 0], 'ob', markersize=5)
    if h is not None:
        ax[2].set_title('main points height')
        ax[2].imshow(image, cmap='gray')
        ax[2].plot(h[:, 1], h[:, 0], 'or', markersize=5)


def hough_transform(edge, show, th):
    image = edge
    h, theta, d = hough_line(edge)  # вычисляем преобразование Хафа от границ изображения
    if show:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()
        ax[1].imshow(image, cmap='gray')
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=th)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[1].plot((0, image.shape[1]), (y0, y1), '-r')
        ax[1].set_xlim((0, image.shape[1]))
        ax[1].set_ylim((image.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Detected lines')
        plt.tight_layout()
        plt.show()
    return hough_line_peaks(h, theta, d, threshold=th)


# пробегааемся по всем линия и удаляем похожие
def remove_similar_lines_with_coeg(lines, coef):
    result_line = []
    indxs = []
    for i in range(len(lines)):
        if i in indxs:
            continue
        a1, d1 = lines[i]
        similar_line = [lines[i]]
        indxs.append(i)
        for j in range(i + 1, len(lines)):
            if j in indxs:
                continue
            a2, d2 = lines[j]
            if abs(d1 - d2) < coef:
                similar_line.append(lines[j])
                indxs.append(j)
        result_line.append(random.choice(similar_line))
    return sorted(result_line, key=lambda x: x[1])


def remove_similar_lines(lines):
    coef = 60
    lines = remove_similar_lines_with_coeg(lines, coef)
    while len(lines) > 3:
        coef += 5
        lines = remove_similar_lines_with_coeg(lines, coef)
    return lines


def get_lines(angles, dists):
    # все линии имеют три направления - 'вертикальные', 'левые', 'правые'
    # выделим такие линии, а так же удалим похожие

    # 3 класса
    df = angles
    df = np.array(df)
    df = df.reshape((-1, 1))
    model = KMeans(n_clusters=3, random_state=42, max_iter=300, n_init=10, verbose=0)
    model.fit(df)
    mean0 = [angles[i] for i in range(len(angles)) if model.labels_[i] == 0]
    mean1 = [angles[i] for i in range(len(angles)) if model.labels_[i] == 1]
    mean2 = [angles[i] for i in range(len(angles)) if model.labels_[i] == 2]
    mean = sorted([(sum(mean0) / len(mean0), 0), (sum(mean1) / len(mean1), 1), (sum(mean2) / len(mean2), 2)])
    r_label = mean[0][1]
    s_label = mean[1][1]
    l_label = mean[2][1]

    straight = [(angles[i], dists[i]) for i in range(len(angles)) if model.labels_[i] == s_label]
    left = [(angles[i], dists[i]) for i in range(len(angles)) if model.labels_[i] == l_label]
    right = [(angles[i], dists[i]) for i in range(len(angles)) if model.labels_[i] == r_label]

    straight = sorted(remove_similar_lines(straight), key=lambda x: x[1])
    left = sorted(remove_similar_lines(left), key=lambda x: x[1])
    right = sorted(remove_similar_lines(right), key=lambda x: x[1])
    return straight, left, right


def draw_lines(image, straight, left, right):
    def draw_one_direct_lines(image, lines, ax, name):
        ax.imshow(image, cmap='gray')
        for angle, dist in lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax.plot((0, image.shape[1]), (y0, y1), '-r')
        ax.set_xlim((0, image.shape[1]))
        ax.set_ylim((image.shape[0], 0))
        ax.set_axis_off()
        ax.set_title(name)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    draw_one_direct_lines(image, straight, ax[0], 'Вертикальные линии')
    draw_one_direct_lines(image, left, ax[1], 'Левые линии')
    draw_one_direct_lines(image, right, ax[2], 'Правые линии')
    plt.tight_layout()
    plt.show()


# устанавилаем порядок между линиями
def get_3_line(lines):
    if len(lines) == 1:
        return lines[0], None, None
    if len(lines) == 2:
        return lines[0], None, lines[1]
    return lines[0], lines[1], lines[2]

# некоторые преобазования С для поиска пересечения линий
def polar_line_to_decart(angle, dist):
    k = tan(angle)
    b = dist / cos(angle)
    return -k, b


def intersect_line_polar(line1, line2):
    if line1 is None or line2 is None:
        return None
    k1, b1 = polar_line_to_decart(*line1)
    k2, b2 = polar_line_to_decart(*line2)
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return x, y


def height_points(straight, left, right):
    s1, s2, s3 = get_3_line(straight)
    l1, l2, l3 = get_3_line(left)
    r1, r2, r3 = get_3_line(right)
    # точки высоты
    a = intersect_line_polar(s1, l1)

    if a is None:
        a = intersect_line_polar(s1, r2)
    if a is None:
        a = intersect_line_polar(l1, r2)
    a1 = intersect_line_polar(s1, r1)
    h = [a, a1]
    if None not in h:
        return np.array(h)

    d = intersect_line_polar(r2, l2)
    if d is None:
        d = intersect_line_polar(r2, s2)
    if d is None:
        d = intersect_line_polar(l2, s2)

    d1 = intersect_line_polar(r1, l3)
    if d1 is None:
        d1 = intersect_line_polar(r1, s2)
    if d1 is None:
        d1 = intersect_line_polar(l3, s2)

    h = [d, d1]
    if None not in h:
        return np.array(h)

    c = intersect_line_polar(r3, l2)
    if c is None:
        c = intersect_line_polar(r3, s3)
    if c is None:
        c = intersect_line_polar(l2, s2)

    c1 = intersect_line_polar(l3, s3)

    h = [c, c1]
    if None not in h:
        return np.array(h)

    return None


def get_width_points(straight, left, right):
    s1, s2, s3 = get_3_line(straight)
    l1, l2, l3 = get_3_line(left)
    r1, r2, r3 = get_3_line(right)
    # точки высоты

    a1 = intersect_line_polar(s1, r1)
    c1 = intersect_line_polar(l3, s3)
    w = [c1, a1]
    if None not in w:
        return np.array(w), False

    a = intersect_line_polar(s1, l1)
    if a is None:
        a = intersect_line_polar(s1, r2)
    if a is None:
        a = intersect_line_polar(l1, r2)

    c = intersect_line_polar(r3, l2)
    if c is None:
        c = intersect_line_polar(r3, s3)
    if c is None:
        c = intersect_line_polar(l2, s2)
    w = [c, a]
    if None not in w:
        return np.array(w), True

    return None, None


def get_box_key_point(img, transform1, transform2, transform3, show=False):
    box_mask = get_box_mask(img, transform1, transform2, transform3, show)
    # теперь оставим только границу куба
    box_edge = get_edges(img, box_mask, show)
    if show:
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Source")
        ax[1].set_title("Box edge")
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(box_edge, cmap="gray")
        plt.show()
    _, angles, dists = hough_transform(box_edge, show, th=100)
    straight, left, right = get_lines(angles, dists)
    if show:
        draw_lines(box_edge, straight, left, right)

    h = height_points(straight, left, right)
    w, flag = get_width_points(straight, left, right)
    if show:
        show_box_point(box_edge, h, w)
    return h, w, flag