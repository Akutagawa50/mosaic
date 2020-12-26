#!/usr/bin/env python3
import cv2

mosaic_img = None
                                                                                                    
def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(img, x, y, width, height, ratio=0.1):
    dst = img.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def main():
    global mosaic_img
    # $ find / 2>/dev/null | grep haarcascade_frontalface_default.xml
    cascade_path = "/mnt/c/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml"

    # Face detection
    img = cv2.imread("kao.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_path)
    facedetect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
    if len(facedetect) > 0:
        # make mosaic
        for x, y, w, h in facedetect:
            mosaic_img = mosaic_area(img, x, y, w, h)
    
    cv2.imwrite("mosaic_kao.jpg", mosaic_img)


if __name__=='__main__':
    main()
