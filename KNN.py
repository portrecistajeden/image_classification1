from data_prep import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

class KNN():
    def __init__(self, trainDir, testDir):
        self.trainDir = trainDir
        self.testDir = testDir

    def createModel(self):
        self.knn = cv2.ml.KNearest_create()

    def trainModel(self):
        self.trainData, self.trainLabels, self.labelsDictionary = load_data_KNN(self.trainDir)
        self.testData, self.testLabels = load_test_KNN(self.testDir)
        self.knn.train(self.trainData, cv2.ml.ROW_SAMPLE, self.trainLabels)

    def evaluateModel(self, k, accPlot, console):
        if k == 0:
            accArray = []
            for k in range(1, 16):
                ret, result, neighbours, dist = self.knn.findNearest(self.testData, k)
                result = result.flatten()
                result = result.astype(np.int)
                acc = 0

                for i in range(len(result)):
                    if result[i] == self.labelsDictionary[self.testLabels[i]]:
                        acc += 1

                accuracy = round(float(acc/len(result)), 2) * 100
                accArray.append(accuracy)

            y = np.arange(1, 16)
            accPlot.plot(y, accArray, label='Accuracy in %')
            accPlot.set_title("Accuracy of KNN algorithm")
            accPlot.set_xlabel("k Value")
            accPlot.set_ylabel("Accuracy %")
            accPlot.set_xlim([1, 15])
            accPlot.legend()
        else:
            ret, result, neighbours, dist = self.knn.findNearest(self.testData, k)

            result = result.flatten()
            result = result.astype(np.int)
            acc = 0

            for i in range(len(result)):
                if result[i] == self.labelsDictionary[self.testLabels[i]]:
                    acc += 1

            accuracy = round(float(acc / len(result)), 2) * 100
            console.append('Accuracy: ' + str(accuracy) + '%')
