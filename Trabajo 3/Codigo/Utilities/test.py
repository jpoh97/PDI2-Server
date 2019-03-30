import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

def func(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (128, 128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from RGB to HSV

    lowerBoundary = np.array([0, 40, 30], dtype="uint8")
    upperBoundary = np.array([43, 255, 254], dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask, 0.5, skinMask, 0.5, 0.0)

    skinMask = cv2.medianBlur(skinMask, 5)

    skin = cv2.bitwise_and(converted2, converted2, mask=skinMask)

    img2 = cv2.Canny(skin, 60, 60)

    surf = cv2.xfeatures2d.SURF_create()

    img2 = cv2.resize(img2, (256, 256))
    kp, des = surf.detectAndCompute(img2, None)

    cv2.waitKey(1)
    return des

clf2 = joblib.load('hands_classifier2.joblib.pkl')

des=func('31.jpg')

cluster_model = MiniBatchKMeans(n_clusters=150)
n_clusters = cluster_model.n_clusters

cluster_model.fit(des)

img_clustered_words = [cluster_model.predict(raw_words) for raw_words in [des]]

X = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

predictions = []

for i in range(0, 100):
    predictions.append(int(clf2.predict(X)))

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

result = np.argmax(np.bincount(np.array(predictions)))

print(letters[result])

