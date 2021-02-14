from data_prep import *

class KNN():
    def __init__(self, trainDir, testDir):
        self.trainData, self.trainLabels = load_data_KNN(trainDir)
        self.testData = load_test_KNN(testDir)
        self.testClassLabels, self.testItemsLabels = getTestLabels(testDir)
        self.trainlabelsStr = getTrainingLabels(trainDir)

    def createModel(self):
        self.knn = cv2.ml.KNearest_create()

    def trainModel(self):
        self.knn.train(self.trainData, cv2.ml.ROW_SAMPLE, self.trainLabels)

    def loadModel(self, path):
        pass

    def saveModel(self, path):
        pass

    def evaluateModel(self, k, accPlot, console):
        if k == 0:
            accArray = []
            for k in range(1, 16):
                ret, result, neighbours, dist = self.knn.findNearest(self.testData, k)
                result = result.flatten()
                result = result.astype(np.int)
                acc = 0
                resultStr = []
                for i in range(len(result)):
                    resultStr.append(self.trainlabelsStr[result[i]])

                for i in range(len(resultStr)):
                    if resultStr[i] == self.testItemsLabels[i]:
                        acc += 1

                accuracy = round(float(acc/len(result)), 2) * 100
                accArray.append(accuracy)

            y = np.arange(15)
            accPlot.plot(y, accArray, label='Accuracy in %')
        else:
            ret, result, neighbours, dist = self.knn.findNearest(self.testData, k)

            result = result.flatten()
            result = result.astype(np.int)
            acc = 0
            resultStr = []
            for i in range(len(result)):
                resultStr.append(self.trainlabelsStr[result[i]])
                console.append('Image ' + str(i) + ': ' + self.trainlabelsStr[result[i]])

            for i in range(len(resultStr)):
                if resultStr[i] == self.testItemsLabels[i]:
                    acc += 1

            accuracy = round(float(acc / len(result)), 2) * 100
            console.append('Accuracy: ' + str(accuracy) + '%')
