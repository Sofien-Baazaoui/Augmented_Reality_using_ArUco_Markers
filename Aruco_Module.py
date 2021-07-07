import cv2
import cv2.aruco as aruco
import numpy as np
import os


def loadAugmentedImages(path):
    myList = os.listdir(path)
    NumofMarkers= len(myList)
    print('Total Number of Markers Detected :', NumofMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics


def findArucoMarkers(img, markersize=6, Totalmarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markersize}X{markersize}_{Totalmarkers}')
    arucoDic = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDic, parameters=arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def ArucoAugmentImage(img, bbox, id, imgAug, drawId=True):
    TopL = bbox[0][0][0], bbox[0][0][1]
    TopR = bbox[0][1][0], bbox[0][1][1]
    BotR = bbox[0][2][0], bbox[0][2][1]
    BotL = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape
    pts1 = np.array([TopL, TopR, BotR, BotL])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgout = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgout = img + imgout

    if drawId:
        cv2.putText(imgout, str(id), TopL, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgout
    

def main():
    cap = cv2.VideoCapture(0)
    augDics = loadAugmentedImages('Markers')
    while True:
        success, img = cap.read()
        arucofound = findArucoMarkers(img)

        # Loop through all the markers and augment each one
        if len(arucofound[0]) != 0:
            for bbox, id in zip(arucofound[0], arucofound[1]):
                if int(id) in augDics.keys():
                    img = ArucoAugmentImage(img, bbox, id, augDics[int(id)])


        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


