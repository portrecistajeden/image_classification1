import random

from data_prep import *

class KNN():
    def __init__(self, trainDir, validationDir):
        self.trainDir = trainDir
        self.predictionsDir = validationDir

    def createModel(self):
        self.knn = cv2.ml.KNearest_create()

    def trainModel(self):
        self.trainData, self.trainLabels, self.labelsDictionary = load_data_KNN(self.trainDir)
        self.predictionData, self.predictionLabels = load_test_KNN(self.predictionsDir)
        self.knn.train(self.trainData, cv2.ml.ROW_SAMPLE, self.trainLabels)

    def evaluateModel(self, k, accPlot, predictGraph, console):
        if k == 0:
            accArray = []
            for k in range(1, 16):
                ret, result, neighbours, dist = self.knn.findNearest(self.predictionData, k)
                result = result.flatten()
                result = result.astype(np.int)
                acc = 0

                for i in range(len(result)):
                    if result[i] == self.labelsDictionary[self.predictionLabels[i]]:
                        acc += 1

                accuracy = round(float(acc/len(result)), 2) * 100
                accArray.append(accuracy)

            y = np.arange(1, 16)
            accPlot.plot(y, accArray, label='Accuracy in %')
            accPlot.set_title("Accuracy of KNN algorithm")
            accPlot.set_xlabel("k Value")
            accPlot.set_ylabel("Accuracy %")
            accPlot.set_xlim([1, 15])
            accPlot.set_xticks(y)
            accPlot.legend()
        else:
            ret, result, neighbours, dist = self.knn.findNearest(self.predictionData, k)

            result = result.flatten()
            result = result.astype(np.int)
            acc = 0

            key_list = list(self.labelsDictionary.keys())

            for i in range(len(result)):
                if result[i] == self.labelsDictionary[self.predictionLabels[i]]:
                    acc += 1

            rand = random.sample(range(0, len(self.predictionData)), k=6)
            ind = 0
            for x in range(2):
                for y in range(3):
                    i = rand[ind]
                    ind += 1

                    img = np.array(self.predictionData[i])
                    img = img.reshape(int(len(img) / 3), -1).T
                    r = img[0].reshape(getTargetSize())
                    g = img[1].reshape(getTargetSize())
                    b = img[2].reshape(getTargetSize())
                    rgb = np.dstack((r, g, b)).astype(np.uint8)

                    predictGraph[x, y].set_title(key_list[result[i]])
                    predictGraph[x, y].imshow(Image.fromarray(rgb, 'RGB'))
                    predictGraph[x, y].axis('off')

            accuracy = round(float(acc / len(result)), 2) * 100
            console.append('Accuracy: ' + str(accuracy) + '%')
