from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, \
    QLineEdit, QLabel, QTextBrowser, QCheckBox, QAction, QMenu, QFrame, QFileDialog, QTabWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from KNN import *
from simpleCNN import *
from customCNN import *

class GraphsTabs(QTabWidget):
    def __init__(self, parent=None):
        super(GraphsTabs, self).__init__(parent)
        self.tab1 = GraphTab()
        self.tab2 = GraphTab()
        self.tab3 = GraphTab()

        self.addTab(self.tab1, "Graph 1")
        self.addTab(self.tab2, "Graph 2")
        self.addTab(self.tab3, "Graph 3")

        self.setMinimumSize(900, 500)

class GraphTab(QWidget):
    def __init__(self):
        super(GraphTab, self).__init__()

        graph = Figure()
        self.graphCanvas = FigureCanvas(graph)
        self.axis = self.graphCanvas.figure.subplots()

        layout = QHBoxLayout()
        layout.addWidget(self.graphCanvas)
        self.setLayout(layout)

        #self.setMinimumHeight(400)

    def draw(self):
        self.graphCanvas.draw()


class Gui(QMainWindow):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

        self.trainDir = "C:\\Users\\piawr\\Desktop\\inżynierk\\minibazka\\small"
        self.testDir = "C:\\Users\\piawr\\Desktop\\inżynierk\\minibazka\\test"

        #menu bar
        self.menuBar = self.menuBar()
        fileMenu = self.menuBar.addMenu('File')

        loadAction = QAction('Load', self)
        loadAction.triggered.connect(self.open_data_window)
        fileMenu.addAction(loadAction)
        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.save_data)
        fileMenu.addAction(saveAction)

        #left panel
        self.algorithmLabel = QLabel('Select first algorithm to compare')
        self.algorithmLabel.setFixedHeight(30)
        self.simpleCNN = QCheckBox('Simple CNN')
        self.simpleCNN.stateChanged.connect(self.uncheckFirst)
        self.knn = QCheckBox('K Nearest Neighbours')
        self.knn.stateChanged.connect(self.uncheckFirst)
        self.customCNN = QCheckBox('Custom CNN')
        self.customCNN.stateChanged.connect(self.uncheckFirst)

        self.trainButton = QPushButton('Execute')
        self.trainButton.clicked.connect(self.trainClick)
        self.trainButton.setFixedWidth(150)
        self.saveModelButton = QPushButton('Save model')
        self.saveModelButton.clicked.connect(self.saveModelClick)
        self.saveModelButton.setFixedWidth(150)
        self.loadModelButton = QPushButton('Load model')
        self.loadModelButton.clicked.connect(self.loadModelClick)
        self.loadModelButton.setFixedWidth(150)
        self.evaluateButton = QPushButton('Evaluate')
        self.evaluateButton.clicked.connect(self.evaluateClick)
        self.evaluateButton.setFixedWidth(150)

        self.kValue = QLineEdit()
        self.kValue.setFixedWidth(150)
        self.kValue.setPlaceholderText('k Value')
        self.kValue.setValidator(QIntValidator())
        self.kValue.setVisible(False)
        self.kValue.setText('0')

        self.epochs = QLineEdit()
        self.epochs.setFixedWidth(150)
        self.epochs.setPlaceholderText('epochs')
        self.epochs.setValidator(QIntValidator())
        self.epochs.setVisible(False)

        #right panel
        self.graphs = GraphsTabs()

        #bottom panel
        self.consolePrint = QTextBrowser()
        self.consolePrint.setReadOnly(True)
        self.consolePrint.setPlaceholderText('console print')
        self.consolePrint.setMinimumHeight(50)
        self.consolePrint.setMaximumHeight(200)
        self.consolePrint.setAlignment(Qt.AlignTop)

        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        self.setWindowTitle('inzynierka')
        self.setBaseSize(1200, 900)
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

        left_part.addWidget(self.kValue)
        left_part.addWidget(self.epochs)

        left_panel.addLayout(left_part)


        right_panel = QGridLayout()
        label1 = QLabel('Graphs for your algoritm')
        right_panel.addWidget(label1, 0, 0)
        right_panel.addWidget(self.graphs, 1, 0)

        top_layout = QHBoxLayout()
        top_layout.addLayout(left_panel)
        top_layout.addLayout(right_panel)
        topFrame = QFrame()
        topFrame.setLayout(top_layout)
        topFrame.setMinimumHeight(700)

        bottom_layout = QHBoxLayout()
        buttons = QVBoxLayout()
        buttons.addWidget(self.loadModelButton)
        buttons.addWidget(self.saveModelButton)
        buttons.addWidget(self.trainButton)
        buttons.addWidget(self.evaluateButton)
        bottom_layout.addLayout(buttons)
        bottom_layout.addWidget(self.consolePrint)
        bottomFrame = QFrame()
        bottomFrame.setLayout(bottom_layout)
        bottomFrame.setFixedHeight(150)

        main_layout = QGridLayout()
        main_layout.addWidget(topFrame, 0, 0)
        main_layout.addWidget(bottomFrame, 1, 0)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def uncheckFirst(self, state):
        if state == Qt.Checked:
            if self.sender() == self.simpleCNN:
                self.kValue.setVisible(False)
                self.epochs.setVisible(True)

                self.knn.setChecked(False)
                self.customCNN.setChecked(False)
            elif self.sender() == self.knn:
                self.kValue.setVisible(True)
                self.epochs.setVisible(False)

                self.simpleCNN.setChecked(False)
                self.customCNN.setChecked(False)
            elif self.sender() == self.customCNN:
                self.kValue.setVisible(False)
                self.epochs.setVisible(True)

                self.knn.setChecked(False)
                self.simpleCNN.setChecked(False)

    def open_data_window(self):
        dialog = LoadDataWindow(self)
        dialog.show()

    def save_data(self):
        print('save')

    def trainClick(self):
        if self.knn.isChecked():
            self.chosenAlgorithm = KNN(self.trainDir, self.testDir)
            self.chosenAlgorithm.train()
            self.chosenAlgorithm.results(int(self.kValue.text()), self.graphs.tab1.axis, self.consolePrint)
            self.topLeftGraphCanvas.draw()

        elif self.simpleCNN.isChecked():
            self.chosenAlgorithm = CNN(self.trainDir, self.testDir, int(self.epochs.text()))
            self.chosenAlgorithm.accGraph(self.graphs.tab1.axis)
            self.graphs.tab1.draw()
            self.chosenAlgorithm.lossGraph(self.graphs.tab2.axis)
            self.graphs.tab2.draw()


        elif self.customCNN.isChecked():
            self.chosenAlgorithm = CustomCNN(self.trainDir, self.testDir, int(self.epochs.text()), self.consolePrint)
            self.chosenAlgorithm.createModel()
            self.chosenAlgorithm.trainModel()
            self.chosenAlgorithm.evaluateModel()
            self.chosenAlgorithm.accGraph(self.TLaxis)
            self.topLeftGraphCanvas.draw()

    def saveModelClick(self):
        pass

    def loadModelClick(self):
        pass

    def evaluateClick(self):
        pass

class LoadDataWindow(QMainWindow):
    def __init__(self, parent):
        super(LoadDataWindow, self).__init__(parent)
        self.setWindowTitle('Load Data')
        self.setFixedSize(400, 200)

        self.training_data_button = QPushButton("Load Training Data")
        self.training_Data_lineedit = QLineEdit()
        self.training_Data_lineedit.setReadOnly(True)
        self.training_data_button.clicked.connect(self.training_data_button_click)

        self.test_data_button = QPushButton("Load Test Data")
        self.test_data_lineedit = QLineEdit()
        self.test_data_lineedit.setReadOnly(True)
        self.test_data_button.clicked.connect(self.test_data_button_click)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.closeButton)

        self.prepare_gui()
        self.show()
    def prepare_gui(self):
        main_layout = QGridLayout()

        main_layout.addWidget(self.training_Data_lineedit, 0, 0, 1, 3)
        main_layout.addWidget(self.training_data_button, 0, 4, 1, 1)
        main_layout.addWidget(self.test_data_lineedit, 1, 0, 1, 3)
        main_layout.addWidget(self.test_data_button, 1, 4, 1, 1)
        main_layout.addWidget(self.close_button, 2, 1, 1, 2)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def closeButton(self):
        self.close()

    def training_data_button_click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select training data directory')
        if file[0]:
            self.training_Data_lineedit.setText(file)
            self.parent().trainDir = file

    def test_data_button_click(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        file = dlg.getExistingDirectory(self, 'Select test data directory')
        if file[0]:
            self.test_data_lineedit.setText(file)
            self.parent().testDir = file