import random
import os
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore")

from skimage import morphology
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from imutils import paths
from sklearn.model_selection import train_test_split

def data(train_list): 
    p=0
    tic = time.perf_counter()

    for img in train_list[:]:
        i= cv2.imread(img)    
        i= cv2.resize(i,(224,224))
        lable= img.split(os.path.sep)[2]

        if (lable=="Benign"):
            b= ('./tmp/prepared_data/benign/'+lable+str(p)+'.png') 
            print("okk")
        if (lable=="[Malignant] Pre-B"):
            b= ('./tmp/prepared_data/PreB/'+lable+str(p)+'.png')  
        if (lable=="[Malignant] Pro-B"):
            b= ('./tmp/prepared_data/ProB/'+lable+str(p)+'.png') 
        if (lable=="[Malignant] early Pre-B"):
            b= ('./tmp/prepared_data/EarlyPreB/'+lable+str(p)+'.png')
        p+=1
        cv2.imwrite(b,i)

            #-------- Segmentation ---------
        i= cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i_lab = cv2.cvtColor(i, cv2.COLOR_RGB2LAB)        #RGB -> LAB
        l,a,b = cv2.split(i_lab)                         
        i2 = a.reshape(a.shape[0]*a.shape[1],1)
        km= KMeans(n_clusters=7, random_state=0).fit(i2)  #Clustring
        p2s= km.cluster_centers_[km.labels_]
        ic= p2s.reshape(a.shape[0],a.shape[1])
        ic = ic.astype(np.uint8)
        r,t = cv2.threshold(ic,141,255 ,cv2.THRESH_BINARY) #Binary Thresholding
        fh = ndi.binary_fill_holes(t)                      #fill holes
        m1 = morphology.remove_small_objects(fh, 200)
        m2 = morphology.remove_small_holes(m1,250)
        m2 = m2.astype(np.uint8)  
        out = cv2.bitwise_and(i, i, mask=m2)

        if (lable=="Benign"):
            b= ('./tmp/prepared_data/benign/'+lable+str(p)+'.png') 
            print('segment')
        if (lable=="[Malignant] Pre-B"):
            b= ('./tmp/prepared_data/PreB/'+lable+str(p)+'.png')  
        if (lable=="[Malignant] Pro-B"):
            b= ('./tmp/prepared_data/ProB/'+lable+str(p)+'.png') 
        if (lable=="[Malignant] early Pre-B"):
            b= ('./tmp/prepared_data/EarlyPreB/'+lable+str(p)+'.png')
        p+=1
        out= cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(b,out)

    toc2 = time.perf_counter() 
    print(f"2917 samples processed in { ((toc2 - tic)/60) } minutes")

def test(test_list):
    p=0

    for img in test_list[:]:
        # print(img)
        i= cv2.imread(img)   
        i= cv2.resize(i,(224,224))
        lable= img.split(os.path.sep)[2]
    
        if (lable=="Benign"):
            b= ('./tmp/prepared_test/benign/'+lable+str(p)+'.png') 
            print("hallo")
        if (lable=="[Malignant] Pre-B"):
            b= ('./tmp/prepared_test/PreB/'+lable+str(p)+'.png')  
        if (lable=="[Malignant] Pro-B"):
            b= ('./tmp/prepared_test/ProB/'+lable+str(p)+'.png') 
        if (lable=="[Malignant] early Pre-B"):
            b= ('./tmp/prepared_test/EarlyPreB/'+lable+str(p)+'.png')
        p+=1
        cv2.imwrite(b,i)

def show(data_list):
    n= 3

    o_img=[]
    a_img=[]
    c_img=[]
    b_img=[]
    m_img=[]
    out_img=[]

    random.seed(865)
    random.shuffle(data_list)

    for img in data_list[:n]:
        i= cv2.imread(img)    
        i= cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i= cv2.resize(i,(300,300))
        o_img.append(i)

        i_lab = cv2.cvtColor(i, cv2.COLOR_RGB2LAB)
        l,a,b = cv2.split(i_lab)
        a_img.append(a)

        i2 = a.reshape(a.shape[0]*a.shape[1],1)
        km= KMeans(n_clusters=7, random_state=0).fit(i2)
        p2s= km.cluster_centers_[km.labels_]
        ic= p2s.reshape(a.shape[0],a.shape[1])
        ic = ic.astype(np.uint8)
        c_img.append(ic)

        r,t = cv2.threshold(ic,141,255 ,cv2.THRESH_BINARY)
        b_img.append(t)   

        fh = ndi.binary_fill_holes(t)   
        m1 = morphology.remove_small_objects(fh, 200)
        m2 = morphology.remove_small_holes(m1,250)
        #m2 = ndi.binary_fill_holes(m2)  
        #m1 = m1.astype(np.uint8)
        #m1_imgs.append(m1)
        m2 = m2.astype(np.uint8)  
        m_img.append(m2)

        out = cv2.bitwise_and(i, i, mask=m2)
        out_img.append(out)

        #corp
        # i= cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        image = i
        img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        # imgBlur = cv2.GaussianBlur(img, (7,7), 0)
        ret, thresh = cv2.threshold(img, 110, 300, 0)
        contuors, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contuors)

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
                    cropped_img=image[y:y+h, x:x+w]
                    img_name= str(i)+".jpg"
                    cv2.imwrite("./img/"+img_name, cropped_img) 
                    # data.append(cropped_img)
        
    return out

    

if __name__ == "__main__":
    data_dir  = './Blood-cell-Cancer-ALL'
    data_list = sorted(list(paths.list_images(data_dir)))
    
    random.seed(88)
    random.shuffle(data_list)

    
    train_list, test_list = train_test_split(data_list, train_size=0.90, shuffle=True, random_state=88)
    
    print('number of testing list -:',len(test_list))
    print('number of training list-:',len(train_list))
    
    print('Number of samples in dataset:',len(list(paths.list_images("./Blood-cell-Cancer-ALL"))),'\n')
    
    print('Number of samples in each class:','\n')
    print("#1 Benign ---------------:", len(list(paths.list_images("./Blood-cell-Cancer-ALL/Benign"))))
    print("#2 Malignant[Early PreB] :", len(list(paths.list_images("./Blood-cell-Cancer-ALL/[Malignant] early Pre-B"))))
    print("#3 Malignant[PreB] ------:", len(list(paths.list_images("./Blood-cell-Cancer-ALL/[Malignant] Pre-B"))))
    print("#4 Malignant[ProB] ------:", len(list(paths.list_images("./Blood-cell-Cancer-ALL/[Malignant] Pro-B"))))
    
    
    # data(train_list)
    d = show(data_list)
