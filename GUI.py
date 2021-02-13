from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, \
    QLineEdit, QLabel, QTextBrowser, QCheckBox, QAction, QMenu, QFrame, QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from KNN import *
from simpleCNN import *
from customCNN import *


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
        self.firstAlgoLabel = QLabel('Select first algorithm to compare')
        self.firstAlgoLabel.setFixedHeight(30)
        self.firstSimpleCNN = QCheckBox('Simple CNN')
        self.firstSimpleCNN.stateChanged.connect(self.uncheckFirst)
        self.firstKNN = QCheckBox('K Nearest Neighbours')
        self.firstKNN.stateChanged.connect(self.uncheckFirst)
        self.firstCustomCNN = QCheckBox('Custom CNN')
        self.firstCustomCNN.stateChanged.connect(self.uncheckFirst)
        self.secondAlgoLabel = QLabel('Select second algorithm to compare')
        self.secondAlgoLabel.setFixedHeight(30)
        self.secondSimpleCNN = QCheckBox('Simple CNN')
        self.secondSimpleCNN.stateChanged.connect(self.uncheckFirst)
        self.secondKNN = QCheckBox('K Nearest Neighbours')
        self.secondKNN.stateChanged.connect(self.uncheckFirst)
        self.secondCustomCNN = QCheckBox('Custom CNN')
        self.secondCustomCNN.stateChanged.connect(self.uncheckFirst)
        self.firstExecute = QPushButton('Execute')
        self.firstExecute.clicked.connect(self.firstExecuteClick)
        self.secondExecute = QPushButton('Execute')
        self.secondExecute.clicked.connect(self.secondExecuteClick)

        self.kValue = QLineEdit()
        self.kValue.setPlaceholderText('k Value')
        self.kValue.setValidator(QIntValidator())
        self.kValue.setVisible(False)
        self.kValue.setText('0')
        self.kValue2 = QLineEdit()
        self.kValue2.setPlaceholderText('k Value')
        self.kValue2.setValidator(QIntValidator())
        self.kValue2.setVisible(False)
        self.kValue2.setText('0')

        self.epochs1 = QLineEdit()
        self.epochs1.setPlaceholderText('epochs')
        self.epochs1.setValidator(QIntValidator())
        self.epochs1.setVisible(False)
        self.epochs2 = QLineEdit()
        self.epochs2.setPlaceholderText('epochs')
        self.epochs2.setValidator(QIntValidator())
        self.epochs2.setVisible(False)

        #right panel
        self.topLeftGraph = Figure()
        self.topLeftGraphCanvas = FigureCanvas(self.topLeftGraph)
        self.TLaxis = self.topLeftGraphCanvas.figure.subplots()

        self.midLeftGraph = Figure()
        self.midLeftGraphCanvas = FigureCanvas(self.midLeftGraph)
        self.MLaxis = self.midLeftGraphCanvas.figure.subplots()

        self.botLeftGraph = Figure()
        self.botLeftGraphCanvas = FigureCanvas(self.botLeftGraph)
        self.BLaxis = self.botLeftGraphCanvas.figure.subplots()

        self.topRightGraph = Figure()
        self.topRightGraphCanvas = FigureCanvas(self.topRightGraph)
        self.TRaxis = self.topRightGraphCanvas.figure.subplots()

        self.midRightGraph = Figure()
        self.midRightGraphCanvas = FigureCanvas(self.midRightGraph)
        self.MRaxis = self.midRightGraphCanvas.figure.subplots()

        self.botRightGraph = Figure()
        self.botRightGraphCanvas = FigureCanvas(self.botRightGraph)
        self.BRaxis = self.botRightGraphCanvas.figure.subplots()

        #bottom panel
        self.consolePrint_left = QTextBrowser()
        self.consolePrint_left.setReadOnly(True)
        self.consolePrint_left.setPlaceholderText('console print')
        self.consolePrint_left.setMinimumSize(100, 200)
        self.consolePrint_left.setAlignment(Qt.AlignTop)

        self.consolePrint_right = QTextBrowser()
        self.consolePrint_right.setReadOnly(True)
        self.consolePrint_right.setPlaceholderText('console print')
        self.consolePrint_right.setMinimumSize(100, 200)
        self.consolePrint_right.setAlignment(Qt.AlignTop)


        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        self.setWindowTitle('inzynierka')
        self.setBaseSize(1200, 900)
        self.setMinimumSize(900, 700)

        left_panel = QHBoxLayout()
        left_panel.setAlignment(Qt.AlignTop)
        left_part = QVBoxLayout()
        left_part.addWidget(self.firstAlgoLabel)
        left_checkboxes = QVBoxLayout()
        left_checkboxes.addWidget(self.firstSimpleCNN)
        left_checkboxes.addWidget(self.firstKNN)
        left_checkboxes.addWidget(self.firstCustomCNN)
        left_part.addLayout(left_checkboxes)
        left_part.addWidget(self.firstExecute)
        left_part.addWidget(self.kValue)
        left_part.addWidget(self.epochs1)
        right_part = QVBoxLayout()
        right_part.addWidget(self.secondAlgoLabel)
        right_checkboxes = QVBoxLayout()
        right_checkboxes.addWidget(self.secondSimpleCNN)
        right_checkboxes.addWidget(self.secondKNN)
        right_checkboxes.addWidget(self.secondCustomCNN)
        right_part.addLayout(right_checkboxes)
        right_part.addWidget(self.secondExecute)
        right_part.addWidget(self.kValue2)
        right_part.addWidget(self.epochs2)

        left_panel.addLayout(left_part)
        left_panel.addLayout(right_part)

        right_panel = QGridLayout()
        label1 = QLabel('Graphs for first algoritm')
        label2 = QLabel('Graphs for second algorithm')
        right_panel.addWidget(label1, 0, 0)
        right_panel.addWidget(label2, 0, 1)
        right_panel.addWidget(self.topLeftGraphCanvas, 1, 0)
        right_panel.addWidget(self.midLeftGraphCanvas, 2, 0)
        right_panel.addWidget(self.botLeftGraphCanvas, 3, 0)
        right_panel.addWidget(self.topRightGraphCanvas, 1, 1)
        right_panel.addWidget(self.midRightGraphCanvas, 2, 1)
        right_panel.addWidget(self.botRightGraphCanvas, 3, 1)

        top_layout = QHBoxLayout()
        top_layout.addLayout(left_panel)
        top_layout.addLayout(right_panel)

        bottom_layout = QGridLayout()
        bottom_layout.addWidget(self.consolePrint_left, 0, 0)
        bottom_layout.addWidget(self.consolePrint_right, 0, 1)

        main_layout = QGridLayout()
        main_layout.addLayout(top_layout, 0, 0)
        main_layout.addLayout(bottom_layout, 1, 0)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def uncheckFirst(self, state):
        if state == Qt.Checked:
            if self.sender() == self.firstSimpleCNN:
                self.kValue.setVisible(False)
                self.epochs1.setVisible(True)

                self.firstKNN.setChecked(False)
                self.firstCustomCNN.setChecked(False)
                self.secondSimpleCNN.setChecked(False)
            elif self.sender() == self.firstKNN:
                self.kValue.setVisible(True)
                self.epochs1.setVisible(False)

                self.firstSimpleCNN.setChecked(False)
                self.firstCustomCNN.setChecked(False)
                self.secondKNN.setChecked(False)
            elif self.sender() == self.firstCustomCNN:
                self.kValue.setVisible(False)
                self.epochs1.setVisible(True)

                self.firstKNN.setChecked(False)
                self.firstSimpleCNN.setChecked(False)
                self.secondCustomCNN.setChecked(False)
            elif self.sender() == self.secondSimpleCNN:
                self.kValue2.setVisible(False)
                self.epochs2.setVisible(True)

                self.secondKNN.setChecked(False)
                self.secondCustomCNN.setChecked(False)
                self.firstSimpleCNN.setChecked(False)
            elif self.sender() == self.secondKNN:
                self.kValue2.setVisible(True)
                self.epochs2.setVisible(False)

                self.secondSimpleCNN.setChecked(False)
                self.secondCustomCNN.setChecked(False)
                self.firstKNN.setChecked(False)
            else:
                self.kValue2.setVisible(False)
                self.epochs2.setVisible(True)

                self.secondKNN.setChecked(False)
                self.secondSimpleCNN.setChecked(False)
                self.firstCustomCNN.setChecked(False)

    def open_data_window(self):
        dialog = LoadDataWindow(self)
        dialog.show()

    def save_data(self):
        print('save')

    def firstExecuteClick(self):
        if self.firstKNN.isChecked():
            self.algorithm1 = KNN(self.trainDir, self.testDir)
            self.algorithm1.train()
            self.algorithm1.results(int(self.kValue.text()), self.TLaxis, self.consolePrint_left)
            self.topLeftGraphCanvas.draw()

        elif self.firstSimpleCNN.isChecked():
            self.algorithm1 = CNN(self.trainDir, self.testDir, int(self.epochs1.text()))
            self.algorithm1.accGraph(self.TLaxis)
            self.topLeftGraphCanvas.draw()
            self.algorithm1.lossGraph(self.MLaxis)
            self.midLeftGraphCanvas.draw()
            # self.algorithm1.predictGraph(self.testDir, self.BLaxis)
            # self.botLeftGraphCanvas.draw()

        elif self.firstCustomCNN.isChecked():
            self.algorithm1 = CustomCNN(self.trainDir, self.testDir, int(self.epochs1.text()), self.consolePrint_left)
            self.algorithm1.accGraph(self.TLaxis)
            self.topLeftGraphCanvas.draw()
            # self.algorithm1.


    def secondExecuteClick(self):
        if self.secondKNN.isChecked():
            self.algorithm2 = KNN(self.trainDir, self.testDir)
            self.algorithm2.train()
            self.algorithm2.results(int(self.kValue2.text()), self.TRaxis, self.consolePrint_right)
            self.topLeftGraphCanvas.draw()

        elif self.secondSimpleCNN.isChecked():
            self.algorithm2 = CNN(self.trainDir, self.testDir, int(self.epochs2.text()))
            self.algorithm2.accGraph(self.TLaxis)
            self.topLeftGraphCanvas.draw()
            self.algorithm2.lossGraph(self.MLaxis)
            self.midLeftGraphCanvas.draw()
            self.algorithm2.predictGraph(self.testDir, self.BRaxis)

        elif self.secondCustomCNN.isChecked():
            self.algorithm2 = CustomCNN(self.trainDir, self.testDir, int(self.epochs2.text()), self.consolePrint_right)


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