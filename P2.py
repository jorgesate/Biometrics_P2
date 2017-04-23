import glob
import cv2
import thinning
import numpy as np


files = glob.glob('huellasFVC04/101_*.tif')

for file_name in files:
    img = cv2.imread(file_name, cv2.COLOR_BGR2GRAY)
    #print np.min(img)
    img2 = img < 120
    img2 = img2.astype(np.uint8) * 255
    print img2.shape
    #print file_name
    cv2.imshow("Orig", img)

    #blur = cv2.GaussianBlur(img, (5, 5), 0)
    #ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 5, 2)
    
    imageK = img.copy()
    imageK = imageK.reshape((imageK.shape[0] * imageK.shape[1], 1))
    clt = KMeans(max_iter=20, n_clusters=3, random_state=7)
    clt.fit(imageK)
    image = np.zeros_like(imageK)
    indices = np.where(clt.labels_ == 1)
    image[indices] = 255
    dst = image.reshape(dst.shape[0], dst.shape[1], 1)

    kernel = np.ones((2, 2), np.uint8)
    prev = adap.copy()
    cv2.imshow("Previo cierre", adap)
    #for _ in range(3):

    adap = cv2.morphologyEx(adap, cv2.MORPH_DILATE, kernel)

    thin = adap.copy()
    thinning.guo_hall_thinning(thin)
    cv2.imshow("Otsu", adap)
    cv2.imshow("skel guo and hall", thin)
    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
