import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# from engine_cancer.show import data

class bld:
    def __init__(self, img):
        self.img = img
        
    def img_dir(self):
        dr = "./Blood_Cancer/"
        dir = os.listdir("./Blood_Cancer/")

        return dir

    def image_resize(self):
        dir = "./Blood_Cancer/"
        Img = self.img_dir()

        image = []
        for i in range(10):
            img = cv2.imread(f"{dir}{Img[i]}")
            # img = cv2.resize(img, (200,200))

            imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGrey, (7,7), 0)

            data = np.asarray(imgBlur)

        return image
    
    def opn(self):
        dr = os.listdir("./Blood_Cancer/")

        img = []
        for i in range(5):
            img.append(dr[i])

        # print(img)

        new_img = []

        for i in range(len(img)):
            Img = cv2.imread(f"./Blood_Cancer/{img[i]}")
            # print(f"./Blood_Cancer/{img[i]}")
            new_img.append(Img)

            img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            # print(img_gray)

            ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
            # print(thresh)
            new_img.append(thresh)
            # print(thresh)

            thresh = 255-thresh

            contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

            img_copy = Img.copy()
            cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(255, 200, 200), thickness=1, lineType=cv2.LINE_AA)
            new_img.append(img_copy)

        # print(new_img)
        return new_img
    
    def detect(self):
        # img = cv2.imread(self.img)
        img = cv2.resize(self.img, (300, 300))

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGrey, (7,7), 0)

        ret, thresh = cv2.threshold(imgBlur, 140, 300, 0)

        imgres = img.copy()

        contuors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_cancer = cv2.drawContours(img, contuors, -1, (0, 0, 255), 5)

        return img_cancer
    
    def crp(self):
        save_file = "./image_save/"
        # img = cv2.imread(self.img)
        img = cv2.resize(self.img, (300,300))

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGrey, (7,7), 0)

        ret, thresh = cv2.threshold(imgBlur, 140, 300, 0)

        # imgres = img.copy()

        contuors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        data = []
        plt.figure(figsize=(10,10))
        for i in range(0,len(contuors)):
            area=cv2.contourArea(contuors[i])
            # print("area",area)

            if(area!=0):
                    x,y,w,h= cv2.boundingRect(contuors[i])
                    x, y = int(x), int(y)
                    w, h = int(w), int(h)

                    print(x, y)
                    print(h ,w)

                    cropped_img=img[y:y+h, x:x+w]

                    # img_name= str(i)+".jpg"
                    # cv2.imwrite(save_file+img_name, cropped_img) 
                    data.append(cropped_img)


        return data
