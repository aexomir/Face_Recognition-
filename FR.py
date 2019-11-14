import cv2 as cv


#preparing image
orig_img = cv.imread('/Users/nic/Downloads/1.jpg')
gray_img = cv.cvtColor(orig_img,cv.COLOR_BGR2GRAY)

#creating CascadeClassifier for FaceRecognition
face_cascade = cv.CascadeClassifier('/anaconda2/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

#detecting faces from grayscale image
detect_face = face_cascade.detectMultiScale(gray_img)

for (column,row,width,height) in detect_face:
    cv.rectangle(
        orig_img,
        (column,row),
        (column+row,width+height),
        (0,255,0),
        2
    )
    #( original image , top-left , bottom-right , color of rectangle , tickness of rectangle )

#showing image
cv.imshow("Process_iamge",orig_img)
cv.waitKey(0)
cv.destroyAllWindows()
