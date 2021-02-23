from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, \
    QLineEdit, QLabel, QTextBrowser, QCheckBox, QAction, QMenu, QFrame, QFileDialog, QTabWidget, QComboBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from KNN import *
from simpleCNN import *
from customCNN import *

class GraphsTabs(QTabWidget):
    def __init__(self, parent=None):
        super(GraphsTabs, self).__init__(parent)
        self.tab1 = GraphTab(False)
        self.tab2 = GraphTab(False)
        self.tab3 = GraphTab(True)

        self.addTab(self.tab1, "Graph 1")
        self.addTab(self.tab2, "Graph 2")
        self.addTab(self.tab3, "Graph 3")

        self.setMinimumSize(800, 500)

class GraphTab(QWidget):
    def __init__(self, predictions):
        super(GraphTab, self).__init__()

        self.graph = Figure()
        self.graphCanvas = FigureCanvas(self.graph)
        if predictions is True:
            self.axis = self.graphCanvas.figure.subplots(2, 3)
        else:
            self.axis = self.graphCanvas.figure.subplots()

        layout = QHBoxLayout()
        layout.addWidget(self.graphCanvas)
        self.setLayout(layout)

        #self.setMinimumHeight(400)

    def draw(self):
        self.graphCanvas.draw()

    def clear(self):
        for x in range(2):
            for y in range(3):
                self.axis[x, y].clear()


class Gui(QMainWindow):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

        self.trainDir = "C:\\Users\\piawr\\Desktop\\inz\\image_classification1\\baza1\\FIDS30"
        self.validationDir = "C:\\Users\\piawr\\Desktop\\inz\\image_classification1\\baza1\\Validation"
        self.predictDir = "C:\\Users\\piawr\\Desktop\\inz\\image_classification1\\baza1\\test"

        self.algorithmFlag = 0

        #menu bar
        self.menuBar = self.menuBar()
        fileMenu = self.menuBar.addMenu('File')

        loadAction = QAction('Load data', self)
        loadAction.triggered.connect(self.open_data_window)
        fileMenu.addAction(loadAction)
        resetAction = QAction('Reset', self)
        resetAction.triggered.connect(self.reset)
        #fileMenu.addAction(resetAction)
        self.menuBar.addAction(resetAction)

        #left panel
        self.algorithmLabel = QLabel('Select first algorithm to compare')
        self.algorithmLabel.setFixedHeight(30)
        self.simpleCNN = QCheckBox('Simple CNN')
        self.simpleCNN.stateChanged.connect(self.uncheckFirst)
        self.knn = QCheckBox('K Nearest Neighbours')
        self.knn.stateChanged.connect(self.uncheckFirst)
        self.customCNN = QCheckBox('Custom CNN')
        self.customCNN.stateChanged.connect(self.uncheckFirst)

        self.createButton = QPushButton('Create model')
        self.createButton.clicked.connect(self.createClick)
        self.createButton.setFixedWidth(150)
        self.createButton.setEnabled(False)
        self.trainButton = QPushButton('Train model')
        self.trainButton.clicked.connect(self.trainClick)
        self.trainButton.setFixedWidth(150)
        self.trainButton.setEnabled(False)
        self.saveModelButton = QPushButton('Save model')
        self.saveModelButton.clicked.connect(self.saveModelClick)
        self.saveModelButton.setFixedWidth(150)
        self.saveModelButton.setEnabled(False)
        self.loadModelButton = QPushButton('Load model')
        self.loadModelButton.clicked.connect(self.loadModelClick)
        self.loadModelButton.setFixedWidth(150)
        self.loadModelButton.setEnabled(False)
        self.evaluateButton = QPushButton('Evaluate')
        self.evaluateButton.clicked.connect(self.evaluateClick)
        self.evaluateButton.setFixedWidth(150)
        self.evaluateButton.setEnabled(False)

        self.parameterLabel = QLabel('Parameter k/epochs:')
        self.parameter = QLineEdit()
        self.parameter.setFixedWidth(150)
        self.parameter.setPlaceholderText('parameter')
        self.parameter.setValidator(QIntValidator())

        self.errorLabel = QLabel('Please choose a parameter')
        self.errorLabel.setStyleSheet("color: red")
        self.errorLabel.setVisible(False)

        self.optimizerLabel = QLabel('Optimizer: ')
        self.optimizer = QComboBox()
        self.optimizer.addItem('adam')
        self.optimizer.addItem('rmsprop')


        #right panel
        self.graphs = GraphsTabs()

        #bottom panel
        self.consolePrint = QTextBrowser()
        self.consolePrint.setReadOnly(True)
        self.consolePrint.setPlaceholderText('console print')
        self.consolePrint.setMinimumHeight(50)
        self.consolePrint.setMaximumHeight(300)
        self.consolePrint.setAlignment(Qt.AlignTop)

        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        self.setWindowTitle('inzynierka')
        self.setBaseSize(1200, 800)
        #self.setMinimumSize(900, 700)

        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)

        left_part = QVBoxLayout()
        left_part.addWidget(self.algorithmLabel)

        checkboxes = QVBoxLayout()
        checkboxes.addWidget(self.simpleCNN)
        checkboxes.addWidget(self.knn)
        checkboxes.addWidget(self.customCNN)
        left_part.addLayout(checkboxes)

        left_part.addWidget(self.parameterLabel)
        left_part.addWidget(self.parameter)
        left_part.addWidget(self.errorLabel)

        cnnParameters = QGridLayout()
        cnnParameters.addWidget(self.optimizerLabel, 0, 0)
        cnnParameters.addWidget(self.optimizer, 0, 1)
        cnnParametersFrame = QFrame()
        cnnParametersFrame.setLayout(cnnParameters)
        left_part.addWidget(cnnParametersFrame)

        left_panel.addLayout(left_part)


        right_panel = QGridLayout()
        label1 = QLabel('Graphs for your algoritm')
        right_panel.addWidget(label1, 0, 0)
        right_panel.addWidget(self.graphs, 1, 0)

        self.graphs.setTabEnabled(1, False)
        self.graphs.setTabEnabled(2, False)

        top_layout = QHBoxLayout()
        top_layout.addLayout(left_panel)
        top_layout.addLayout(right_panel)
        topFrame = QFrame()
        topFrame.setLayout(top_layout)
        topFrame.setMinimumHeight(600)

        bottom_layout = QHBoxLayout()
        buttons = QVBoxLayout()
        buttons.addWidget(self.createButton)
        buttons.addWidget(self.trainButton)
        buttons.addWidget(self.evaluateButton)
        buttons.addWidget(self.loadModelButton)
        buttons.addWidget(self.saveModelButton)
        bottom_layout.addLayout(buttons)
        bottom_layout.addWidget(self.consolePrint)
        bottomFrame = QFrame()
        bottomFrame.setLayout(bottom_layout)
        bottomFrame.setMaximumHeight(300)

        main_layout = QGridLayout()
        main_layout.addWidget(topFrame, 0, 0)
        main_layout.addWidget(bottomFrame, 1, 0)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def uncheckFirst(self, state):
        if state == Qt.Checked:
            if self.sender() == self.simpleCNN:
                self.algorithmFlag = 1

                self.parameterLabel.setText('Epochs:')

                self.knn.setChecked(False)
                self.customCNN.setChecked(False)

                self.loadModelButton.setEnabled(True)
                self.createButton.setEnabled(True)

                self.graphs.setTabEnabled(0, True)
                self.graphs.setTabEnabled(1, True)
                self.graphs.setTabEnabled(2, True)

            elif self.sender() == self.knn:
                self.algorithmFlag = 2

                self.parameterLabel.setText('Parameter k:')

                self.simpleCNN.setChecked(False)
                self.customCNN.setChecked(False)

                self.loadModelButton.setEnabled(False)
                self.createButton.setEnabled(True)

                self.graphs.setTabEnabled(0, True)
                self.graphs.setTabEnabled(1, False)
                self.graphs.setTabEnabled(2, True)

            else:
                self.algorithmFlag = 3

                self.parameterLabel.setText('Epochs:')

                self.knn.setChecked(False)
                self.simpleCNN.setChecked(False)

                self.loadModelButton.setEnabled(True)
                self.createButton.setEnabled(True)

                self.graphs.setTabEnabled(0, True)
                self.graphs.setTabEnabled(1, True)
                self.graphs.setTabEnabled(2, True)

        elif state == Qt.Unchecked:
            self.trainButton.setEnabled(False)
            self.loadModelButton.setEnabled(False)
            self.createButton.setEnabled(False)
            self.simpleCNN.setEnabled(True)
            self.knn.setEnabled(True)
            self.customCNN.setEnabled(True)


    def open_data_window(self):
        dialog = LoadDataWindow(self)
        dialog.show()

    def reset(self):
        self.chosenAlgorithm = None

        self.parameterLabel.setText('Parameter k/epochs:')

        self.simpleCNN.setEnabled(True)
        self.knn.setEnabled(True)
        self.customCNN.setEnabled(True)

        self.simpleCNN.setChecked(False)
        self.knn.setChecked(False)
        self.customCNN.setChecked(False)

        self.loadModelButton.setEnabled(False)
        self.evaluateButton.setEnabled(False)
        self.saveModelButton.setEnabled(False)

        self.graphs.tab1.axis.clear()
        self.graphs.tab1.draw()
        self.graphs.tab2.axis.clear()
        self.graphs.tab2.draw()
        self.graphs.tab3.clear()
        self.graphs.tab3.draw()

        self.parameter.setText("")
        self.consolePrint.setText("")

        self.graphs.setTabEnabled(0, True)
        self.graphs.setTabEnabled(1, False)
        self.graphs.setTabEnabled(2, False)

    def createClick(self):
        if self.simpleCNN.isChecked():
            self.knn.setEnabled(False)
            self.customCNN.setEnabled(False)

            self.chosenAlgorithm = CNN(self.trainDir, self.validationDir, self.predictDir, self.optimizer.currentText(), self.consolePrint)

        elif self.knn.isChecked():
            self.simpleCNN.setEnabled(False)
            self.customCNN.setEnabled(False)

            self.chosenAlgorithm = KNN(self.trainDir, self.validationDir)

        elif self.customCNN.isChecked():
            self.simpleCNN.setEnabled(False)
            self.knn.setEnabled(False)

            self.chosenAlgorithm = CustomCNN(self.trainDir, self.validationDir, self.predictDir, self.optimizer.currentText(), self.consolePrint)

        self.chosenAlgorithm.createModel()
        self.trainButton.setEnabled(True)



    def trainClick(self):
        if self.simpleCNN.isChecked():
            if self.parameter.text() != "":
                self.errorLabel.setVisible(False)
                self.chosenAlgorithm.trainModel(int(self.parameter.text()))

                self.chosenAlgorithm.accGraph(self.graphs.tab1.axis)
                self.graphs.tab1.draw()
                self.chosenAlgorithm.lossGraph(self.graphs.tab2.axis)
                self.graphs.tab2.draw()

                self.saveModelButton.setEnabled(True)
                self.evaluateButton.setEnabled(True)
            else:
                self.errorLabel.setVisible(True)

        elif self.knn.isChecked():
            self.chosenAlgorithm.trainModel()

            self.saveModelButton.setEnabled(False)
            self.loadModelButton.setEnabled(False)
            self.evaluateButton.setEnabled(True)

        elif self.customCNN.isChecked():
            if self.parameter.text() != "":
                self.errorLabel.setVisible(False)
                self.chosenAlgorithm.trainModel(int(self.parameter.text()))

                self.chosenAlgorithm.accGraph(self.graphs.tab1.axis)
                self.graphs.tab1.draw()
                self.chosenAlgorithm.lossGraph(self.graphs.tab2.axis)
                self.graphs.tab2.draw()

                self.saveModelButton.setEnabled(True)
                self.evaluateButton.setEnabled(True)
            else:
                self.errorLabel.setVisible(True)

    def evaluateClick(self):
        if self.algorithmFlag == 2:
            self.chosenAlgorithm.evaluateModel(int(self.parameter.text()),
                                               self.graphs.tab1.axis,
                                               self.graphs.tab3.axis,
                                               self.consolePrint)
            self.graphs.tab1.draw()
            self.graphs.tab3.draw()
        else:
            self.chosenAlgorithm.evaluateModel()
            self.chosenAlgorithm.predictGraph(self.graphs.tab3.axis)
            self.graphs.tab3.draw()

    def saveModelClick(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select directory to save the model')
        if file != "":
            savePath = file

        self.chosenAlgorithm.saveModel(savePath)

    def loadModelClick(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select directory to save the model')
        if file != "":
            loadPath = file

        if self.algorithmFlag == 1:
            self.chosenAlgorithm = CNN(self.trainDir, self.validationDir)
            self.chosenAlgorithm.loadModel(loadPath)
        elif self.algorithmFlag == 3:
            self.chosenAlgorithm = CustomCNN(self.trainDir, self.validationDir, self.consolePrint)
            self.chosenAlgorithm.loadModel(loadPath)

        self.saveModelButton.setEnabled(True)
        self.trainButton.setEnabled(True)
        self.evaluateButton.setEnabled(True)


class LoadDataWindow(QMainWindow):
    def __init__(self, parent):
        super(LoadDataWindow, self).__init__(parent)
        self.setWindowTitle('Load Data')
        self.setFixedSize(400, 200)

        self.training_data_button = QPushButton("Load Training Data")
        self.training_Data_lineedit = QLineEdit()
        self.training_Data_lineedit.setReadOnly(True)
        self.training_Data_lineedit.setText(self.parent().trainDir)
        self.training_data_button.clicked.connect(self.training_data_button_click)

        self.validation_data_button = QPushButton("Load Validation Data")
        self.validation_data_lineedit = QLineEdit()
        self.validation_data_lineedit.setReadOnly(True)
        self.validation_data_lineedit.setText(self.parent().validationDir)
        self.validation_data_button.clicked.connect(self.test_data_button_click)

        self.predict_data_button = QPushButton("Load Predict Data")
        self.predict_data_lineedit = QLineEdit()
        self.predict_data_lineedit.setReadOnly(True)
        self.predict_data_lineedit.setText(self.parent().predictDir)
        self.predict_data_button.clicked.connect(self.predict_data_button_click)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.closeButton)

        self.prepare_gui()
        self.show()
    def prepare_gui(self):
        main_layout = QGridLayout()

        main_layout.addWidget(self.training_Data_lineedit, 0, 0, 1, 3)
        main_layout.addWidget(self.training_data_button, 0, 4, 1, 1)
        main_layout.addWidget(self.validation_data_lineedit, 1, 0, 1, 3)
        main_layout.addWidget(self.validation_data_button, 1, 4, 1, 1)
        main_layout.addWidget(self.predict_data_lineedit, 2, 0, 1, 3)
        main_layout.addWidget(self.predict_data_button, 2, 4, 1, 1)
        main_layout.addWidget(self.close_button, 3, 1, 1, 2)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def closeButton(self):
        self.close()

    def training_data_button_click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select training data directory')
        if file != "":
            self.training_Data_lineedit.setText(file)
            self.parent().trainDir = file

    def test_data_button_click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select validation data directory')
        if file != "":
            self.validation_data_lineedit.setText(file)
            self.parent().validationDir = file

    def predict_data_button_click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select predict data directory')
        if file != "":
            self.predict_data_lineedit.setText(file)
            self.parent().predictDir = file