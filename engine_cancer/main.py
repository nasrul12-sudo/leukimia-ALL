import cv2 
import show
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from imutils import paths
from scipy import ndimage as ndi
from skimage import morphology

class crp:
    def __init__(self):
        self.img = "./F1.jpg"
        # self.o_img = []

    def crop(self):
        i= cv2.imread(self.img)    
        i= cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i= cv2.resize(i,(300,300))
        # self.o_img.append(i)

        i_lab = cv2.cvtColor(i, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(i_lab)
        # a_img.append(a)

        i2 = a.reshape(a.shape[0]*a.shape[1],1)
        km= KMeans(n_clusters=7, random_state=0).fit(i2)
        p2s= km.cluster_centers_[km.labels_]
        ic= p2s.reshape(a.shape[0],a.shape[1])
        ic = ic.astype(np.uint8)
        # c_img.append(ic)/

        r,t = cv2.threshold(ic,141,255 ,cv2.THRESH_BINARY)
        # b_img.append(t)  

        fh = ndi.binary_fill_holes(t)   
        m1 = morphology.remove_small_objects(fh, 200)
        m2 = morphology.remove_small_holes(m1,250)
        # m2 = ndi.binary_fill_holes(m2)  
        #m1 = m1.astype(np.uint8)
        #m1_imgs.append(m1)
        m2 = m2.astype(np.uint8)  
        # m_img.append(m2)

        out = cv2.bitwise_and(i, i, mask=m2)

        contuors, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        plt.imshow(out)
        plt.show()
        
        # plt.figure(figsize=(10,10))
        # for i in range(0,len(contuors)):
        #     area=cv2.contourArea(contuors[i])
        #     # print("area",area)

        #     if(area!=0):
        #             x,y,w,h= cv2.boundingRect(contuors[i])
        #             x, y = int(x), int(y)
        #             w, h = int(w), int(h)

        #             print(x, y)
        #             print(h ,w)

                    # cropped_img=i[y:y+h, x:x+w]
                    # print(cropped_img)

                    # img_name= str(i)+".jpg"
                    # cv2.imwrite(save_file+img_name, cropped_img) 
                    # data.append(cropped_img)

        # plt.imshow(contuors)
        # plt.show()
        return out
    
    def detect(self):
        img = self.crop()

        imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGrey, (5,5), 0)
        ret, thresh = cv2.threshold(imgBlur, 140, 300, 0)

        # imgres = img.copy()

        contuors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(contuors)

        plt.imshow(thresh)
        plt.show()

c = crp()
c.crop()
# c.detect()
