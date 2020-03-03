import os
import cv2
import sys
import json
import random
import numpy as np

# from skimage.transform import *
# import skimage.transform as tf


mask_root = "masks_new/"


label_root = "E:/python/Web Scrapping/Luggage_data_set/Luggage_data_set"
point_export = "E:/python/Web Scrapping/Luggage_data_set/annotations"


img_root = "img_root/"


if not os.path.exists(point_export):
    os.mkdir(point_export)

W, H = 3264, 2448

car_idx = 1
comp_idx = 4
view_idx = 0
overwrite = False

FILLER = {
    "AL": "2G",
    "WR": "2G",
    "SW": "10",
    "DZ": "15"
}
CARS = [
    "AL",
    "WR",
    "SW",
    "DZ"
]
COMPONENTS = [
    "CLFF",
    "CFBU",
    "CBOO",
    "CRFF",
    "CLFD",
    "CRFD"
]
VIEWS = [
    "L",
    "R"
]

car = CARS[car_idx]
view = VIEWS[view_idx]
comp = COMPONENTS[comp_idx]
fill = FILLER[CARS[car_idx]]


def paint_partial(event, x, y, flags, param):
    global image
    global points
    if event == cv2.EVENT_FLAG_LBUTTON:
        c_color = colors[int(len(points)/2)]
        points.append((x, y))
        print("({}, {}) point added".format(x, y))
        cv2.circle(overlay, (x, y), 15, c_color, -1)
        image = cv2.putText(overlay, str(len(points)),  (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,25,100), 2, cv2.LINE_AA)


def prep_transform():
    cen = cen_fill_image.copy()
    alt = alt_fill_image.copy()
    for each in points[::2]:
        cen = cv2.circle(cen, (each[0], each[1]), 10, (0, 0, 255), -1)
    for each in points[1::2]:
        alt = cv2.circle(alt, (each[0], each[1]), 10, (255, 0, 0), -1)

    p1 = points[::2]
    p2 = points[1::2]

    p1 = np.float32(p1)
    p2 = np.float32(p2)
    p2[..., 0] -= W

    tform = PiecewiseAffineTransform()
    tform.estimate(p2, p1)
    # transform image
    im_transform = tf.warp(alt, tform.inverse)
    im_transform *= 255

    cen = np.uint8(cen)
    im_transform = np.uint8(im_transform)
    im_transform = cv2.bitwise_and(im_transform, cen_mask_image)
    cen = cv2.bitwise_and(cen, cen_mask_image)
    for each in points[::2]:
        im_transform = cv2.circle(
            im_transform, (each[0], each[1]), 10, (0, 0, 255), 5)

    im_transform = cv2.rectangle(im_transform, (0, 0), (W, H), (0, 255, 0), 10)
    cen = cv2.rectangle(cen, (0, 0), (W, H), (0, 255, 0), 10)

    return np.concatenate((cen, im_transform), axis=1)


# mask_root = "masks_new/"
# label_root = "masks_concat_new/"
# point_export = "masks_points_new/"
# img_root = "car_images/"


print(mask_root + "{}_{}_{}_C_1.jpg".format(car, fill, comp, view))
# print(cen_mask_image.shape)
# cen_mask_image = cv2.resize(cen_mask_image, (W, H))
# cen_fill_image = cv2.resize(cen_fill_image, (W, H))
# alt_fill_image = cv2.resize(alt_fill_image, (W, H))
# overlay = np.zeros_like(label_image)

def get_previous_points(pt_file_path):
    points = []
    try:
        with open(pt_file_path, "r") as f:
            d = json.loads(f.read())
        points = d['points']
    except:
        pass

    if overwrite or points is None:
        points = []

    for i, each in enumerate(points):
        c_color = colors[i]
        cv2.circle(overlay, (each[0], each[1]), 15, c_color, -1)
        image = cv2.putText(overlay, str(i+1),  (each[0], each[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,25,100), 2, cv2.LINE_AA)

    return points

def color_gen(): return (random.randint(0, 255),
                         random.randint(0, 255), random.randint(0, 255))


colors = [color_gen() for i in range(200)]




def main():
    image_files = os.listdir(label_root)
    points_files = os.listdir(point_export)
    # filterd_image_files = [i for i in image_files if i.split('.')[0] not in [k.split('.')[0] for k in points_files]]
    filterd_image_files = image_files
    print("files to process : " , len(filterd_image_files))
    for iii, f in enumerate(filterd_image_files):
        print(iii,f)
        global label_image

        label_image = cv2.imread(label_root + '/' + f)
        pt_file_path = point_export + "/" + f.split('.')[0] + ".json"
        print(label_image.shape)
        
        cen_mask_image = cv2.imread(mask_root + "{}_{}_{}_C_1.jpg".format(
            car, fill, comp, view
        ))
        # cen_mask_image.shape
        alt_fill_image = cv2.imread(img_root + "{}_{}_{}_{}_1.jpg".format(
            car, fill, comp, view
        ))
        cen_fill_image = cv2.imread(img_root + "{}_{}_{}_C_1.jpg".format(
            car, fill, comp, view
        ))
        # cen_mask_image = cv2.resize(cen_mask_image, (W, H))
        # cen_fill_image = cv2.resize(cen_fill_image, (W, H))
        # alt_fill_image = cv2.resize(alt_fill_image, (W, H))
        xyz(pt_file_path)
    # Create a window and set Mousecallback to a function for that window

    # if old_t is not None:
    #         cv2.imshow('transformed', old_t)
    # else:
    #     cv2.imshow('transformed', np.zeros((H, W, 3)))
    # Do until esc pressed

    cv2.destroyAllWindows()


def xyz(pt_file_path):
    global points
    global label_image
    global overlay
    overlay = np.zeros_like(label_image)
    cv2.namedWindow('draw', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('transformed', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('draw', paint_partial)
    points = get_previous_points(pt_file_path)
    while True:
        out = cv2.addWeighted(label_image, 0.7, overlay, 1, 1)
        cv2.imshow('draw', out)
        k = cv2.waitKey(1)

        if k == 27 or k == ord('n'):
            print('next image')
            break

        elif k == ord('u'):
            if len(points) > 0:
                cv2.circle(overlay, (points[-1][0],
                                     points[-1][1]), 15, (0, 0, 0), -1)
                p = points.pop()
                print("{} point removed".format(p))
                continue

        elif k == ord('d'):
            res_dict = {
                'points': points,
                'is_valid' : True
            }
            with open(pt_file_path, 'w') as fp:
                json.dump(res_dict, fp)
            print('saving points')
            break

        elif k == ord('x'):
            res_dict = {
                'points': None,
                'is_valid' : False
            }
            with open(pt_file_path, 'w') as fp:
                json.dump(res_dict, fp)
            print('skip image')
            break

        elif k == ord('s'):
            sys.exit()

        # elif k == ord('t'):
        #     if len(points) % 2 != 0 or len(points) < 4:
        #         continue
        #     else:
        #         disp = prep_transform()
        #         cv2.imshow('transformed', disp)


if __name__ == '__main__':
    main()
