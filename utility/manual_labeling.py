import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

base = "./people"
clicked = False
X, Y = None, None


def store_drawback(event, x, y, flags, params):
    # params: [filename]
    global clicked, X, Y
    if event == cv.EVENT_LBUTTONDOWN:
        clicked = True
        X, Y = x, y

        with open(f"{path}", "w") as wf:
            print(f"{x} {y}\n")
            # X, Y = x, y
            wf.write(f"{x} {y}\n")


def get_bounding_box(img, mask, lower_pos: tuple):
    x, y = lower_pos
    # print(x, y)
    _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    yy, xx = np.where(mask[: y, :] > 0)
    x1, x2 = np.min(xx), np.max(xx)
    y1, y2 = np.min(yy), y
    face_slice = img[y1 : y2, x1 : x2, :]
    # print(x1, x2, y1, y2)

    return face_slice, x1, y1, x2, y2


def read_file(filename: str):
    with open(filename, "r") as rf:
        line = rf.readline().strip()
        x, y = map(int, line.split(" "))

    return x, y


if __name__ == "__main__":
    # print(os.getcwd())
    dirs = ["masked", "unmasked", "testing"]

    for dir_iter in dirs:
        count = 0
        for root, _, files in os.walk(os.path.join(base, dir_iter)):
            for file in files:
                count += 1
                print(f"have processed {count}/{len(files)} images")

                if file.find("mask") != -1:
                    continue

                ind = file.find(".png")
                if ind == -1:
                    continue

                filename = file[:ind] + ".txt"
                path = os.path.join(root, filename)
                img_path = os.path.join(root, file)
                img = cv.imread(img_path)
                mask_path = os.path.join(root, "mask_" + file)
                img_mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                print(f"{img_path}, {mask_path}")

                try:
                    if os.path.isfile(path):
                        print(f"file: {path} already processed")
                        x, y = read_file(path)
                        img_face, _, _, _, _ = get_bounding_box(img, img_mask, (x, y))
                        while True:
                            cv.imshow("face", img_face)

                            k = cv.waitKey(-1) & 0xFF
                            if k == ord('q') or k == 27:
                                break
                        cv.destroyAllWindows()
                        continue

                    cv.namedWindow("img")
                    cv.setMouseCallback("img", store_drawback)
                    while True:
                        cv.imshow("img", img)
                        cv.imshow("mask", img_mask)

                        k = cv.waitKey(-1) & 0xFF
                        if k == ord('q') or k == 27:
                            break

                    while True:
                        x, y = read_file(path)
                        img_face, _, _, _, _ = get_bounding_box(img, img_mask, (X, Y))
                        cv.imshow("face", img_face)

                        k = cv.waitKey(-1) & 0xFF
                        if k == ord('q') or k == 27:
                            break

                    # print(os.path.join(root, "face_" + img_path))
                    cv.imwrite(os.path.join(root, "face_" + file), img_face)

                    # answer = input("Are you satisfied with the click: (Y/N)")
                    # if answer == "Y":
                    #     # with open(f"{path}", "w") as wf:
                    #     #     print(f"{x} {y}\n")
                    #     #     X, Y = x, y
                    #     #     wf.write(f"{x} {y}\n")
                    #     cv.imwrite(os.path.join(root, "face_" + img_path), img_face)
                    # elif answer == "N":
                    #     os.remove(path)

                    if clicked:
                        cv.destroyAllWindows()
                        clicked = False

                except Exception as e:
                    with open("outliers.txt", "a") as wf:
                        wf.write(f"{file}\n")

