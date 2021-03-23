from math import sqrt
import matplotlib.pyplot as plt


def get_l2(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def two_point_to_line(p1, p2):
    k = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - k * p1[0]
    return k, b


def intersect_line_decart(line1, line2):
    k1, b1 = line1
    k2, b2 = line2
    x = (b2 - b1) / (k1 - k2)
    y = k1*x+b1
    return x, y


def get_f_hor(b, e, c, g, img, show):
    line1 = two_point_to_line(b, e)
    line2 = two_point_to_line(c, g)
    F_hor = intersect_line_decart(line1, line2)
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot([F_hor[0], b[0]], [F_hor[1], b[1]], 'r')
        plt.plot([F_hor[0], c[0]], [F_hor[1], c[1]],'r')
        plt.title('Нахождение F_hor')
        plt.plot(F_hor[0], F_hor[1],'ob', markersize=5)
        plt.tight_layout()
        plt.show()
    return F_hor


def get_f_ver(a, b, c, d, img, show):
    line1 = two_point_to_line(a, b)
    line2 = two_point_to_line(c, d)
    F_ver = intersect_line_decart(line1, line2)
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot([F_ver[0], a[0]],[F_ver[1], a[1]], 'r')
        plt.plot([F_ver[0], d[0]] , [F_ver[1], d[1]],'r')
        plt.plot(F_ver[0], F_ver[1],'ob', markersize=5)
        plt.title('Нахождение F_ver')
        plt.tight_layout()
        plt.show()
    return F_ver


def chech_height(a, d, b1, c1, h1, h2, F_ver, F_hor, img, show):
    # проверка, пройдет ли короба по высоте
    line1 = two_point_to_line(h2, F_hor)
    line2 = two_point_to_line(a, d)
    p1 = intersect_line_decart(line1, line2)

    line3 = two_point_to_line(p1, F_ver)
    line4 = two_point_to_line(h1, F_hor)
    p2 = intersect_line_decart(line3, line4)

    k, b = two_point_to_line(b1, c1)
    x, y = p2[0], p2[1]
    if k*x + b > y:
        ans1 = False
    else:
        ans1 = True
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot([h2[0], F_hor[0]], [h2[1], F_hor[1]], 'r')
        plt.plot([h1[0], F_hor[0]], [h1[1], F_hor[1]],'r')
        plt.plot(p1[0], p1[1],'ob', markersize=5)
        plt.plot(p2[0], p2[1],'ob', markersize=5)
        plt.plot([p1[0],  p2[0]], [p1[1],  p2[1]],'b')
        plt.tight_layout()
        plt.show()
    return ans1

def check_width(a, d, a1, d1, w1, w2, F_hor, img, show):
    # проверка пройдет ли коробка по ширине
    line1 = two_point_to_line(a, d)
    line2 = two_point_to_line(w1, F_hor)
    line3 = two_point_to_line(w2, F_hor)

    p1 = intersect_line_decart(line1, line2)
    p2 = intersect_line_decart(line1, line3)
    real_dist = get_l2(a1, d1)
    dist = get_l2(p1, p2)

    if dist >= real_dist:
        ans2 = False
    else:
        ans2 = True
    if show:
        plt.imshow(img, cmap='gray')
        plt.plot([w2[0], F_hor[0]],[w2[1], F_hor[1]], 'r')
        plt.plot([w1[0], F_hor[0]], [w1[1], F_hor[1]],'r')
        plt.plot(p1[0], p1[1],'ob', markersize=5)
        plt.plot(p2[0], p2[1],'ob', markersize=5)
        plt.plot([p1[0],  p2[0]], [p1[1],  p2[1]],'b')
        plt.tight_layout()
        plt.show()
    return ans2


def get_answer(transform1, transform2, transform3, h, w, flag, img, show):
    h1 = [h[0][1], h[0][0]]
    h2 = [h[1][1], h[1][0]]

    w1 = [w[0][1], w[0][0]]
    w2 = [w[1][1], w[1][0]]
    if flag:
        dist = get_l2(h1, h2)
        w1[1] += dist
        w2[1] += dist

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
    F_hor = get_f_hor(b, e, c, g, img, show)
    F_ver = get_f_ver(a, b, c, d, img, show)
    ans1 = chech_height(a, d, b1, c1, h1, h2, F_ver, F_hor, img, show)
    ans2 = check_width(a, d, a1, d1, w1, w2, F_hor, img, show)
    return ans1 and ans2