#%%
import cv2
import numpy as np

def photo_stack():
    path = "./output/"        

    img1 = cv2.imread(path+"pyramid-egypt.jpg")
    img2 = cv2.imread(path+"glico-japan.jpg")
    add1 = np.concatenate([img1, img2],axis=0)
    cv2.imwrite("add1.jpg",add1)

    img3 = cv2.imread(path+"freedom-USA.jpg")
    img4 = cv2.imread(path+"pisa-italy.jpg")
    add2 = np.concatenate([img3, img4],axis=0)
    cv2.imwrite("add2.jpg",add2)

    add1 = cv2.imread("add1.jpg")
    add2 = cv2.imread("add2.jpg")
    result1 = np.concatenate([add1, add2],axis=0)
    # cv2.imwrite("result1.jpg",result1)
    return result1

# result1 = photo_stack()
# cv2.imwrite("result3.jpg",result1)
# img = cv2.imread("result3.jpg",)
# # resized = cv2.resize(img, (500,600))

# cv2.imshow("result3.jpg",img)
# cv2.waitKey(0)

# %%
