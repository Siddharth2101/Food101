import numpy as np
import cv2

def convolution(img, kernel):
    row,col=img.shape
    krow,kcol=kernel.shape
    orow,ocol=row-krow+1,col-kcol+1
    final=np.zeros((orow, ocol),dtype=np.float64)
    for i in range(orow):
        for j in range(ocol):
            region=img[i:i+krow,j:j+kcol]
            temp=np.sum(region*kernel)
            final[i,j]=temp        
    final = np.clip(final, 0, 255).astype(np.uint8)
    return final


def convolution(img, kernel):
    row,col=img.shape
    krow,kcol=kernel.shape
    orow,ocol=row-krow+1,col-kcol+1
    final=np.zeros((orow, ocol),dtype=np.float64)
    for i in range(orow):
        for j in range(ocol):
            region=img[i:i+krow,j:j+kcol]
            temp=np.sum(region*kernel)
            final[i,j]=temp        
    final = np.clip(final, 0, 255).astype(np.uint8)
    return final




img=cv2.imread('image.webp', cv2.IMREAD_GRAYSCALE)
flipped_img = cv2.flip(img, 1)

k_sh=np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
k_bl=np.ones((3,3))/9
k_ed=np.array([[5, 0, -5],[5, 0, -5],[5, 0, -5]])
img1=convolution(img, k_sh)
img2=convolution(img, k_bl)
img3_1=convolution(img, k_ed)
img_fc=convolution(flipped_img, k_ed) #flipping image to highlight with white edges everywhere
img3_2=cv2.flip(img_fc, 1)
img3=img3_1+img3_2+img[1:len(img)-1,1:len(img[0])-1]

cv2.imshow("Original Image", img)
cv2.imshow("Sharped Image",img1)
cv2.imshow("Blurry",img2)
cv2.imshow("Highlighted Edge Image",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()