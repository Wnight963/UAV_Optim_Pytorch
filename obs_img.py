import numpy as np
import matplotlib.pyplot as plt
import cv2

def set_value_in_img(img, pos_x, pos_y, value):
    """
    draw 3*3 rectangle in img
    :param img:
    :param pos_x:
    :param pos_y:
    :param value:
    :return:
    """
    img_obs_size_x = 300
    img_obs_size_y = 300

    img[pos_x, pos_y] = value


if __name__ == '__main__':
    img_obs_size_x = 300
    img_obs_size_y = 300
    # 个体img：所有己方单位位置
    obs_img = np.full((img_obs_size_x, img_obs_size_y, 3), 255, dtype=np.int32)
    # print(obs_img)
    a = plt.imshow(obs_img)
    # plt.show(a)

    for i in range(300):
        set_value_in_img(obs_img, 150, 150, [48, 158, 67])
    # print(obs_img)
    b = plt.imshow(obs_img, filternorm=1, filterrad=4, interpolation='lanczos')
    plt.show(b)
    print(obs_img[150][150])