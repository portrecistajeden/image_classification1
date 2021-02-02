import cv2
import numpy as np
from data_prep import load_data_KNN, load_test_KNN

class KNN():
    def __init__(self, trainDir, testDir):
        self.trainData, self.trainLabels = load_data_KNN(trainDir)
        self.testData = load_test_KNN(testDir)
        self.knn = cv2.ml.KNearest_create()

    def train(self):
        self.knn.train(self.trainData, cv2.ml.ROW_SAMPLE, self.trainLabels)

    def results(self, k):
        ret, result, neighbours, dist = self.knn.findNearest(self.testData, k)
        print('return: ', ret)
        print('result; ', result)
        print('neighbours: ', neighbours)
        print('dist: ', dist)

        # matches = result == self.testLabels
        # correct = np.count_nonzero(matches)
        # accuracy = correct * 100.0 / result.size
        # print(accuracy)