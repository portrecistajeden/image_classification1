from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, \
    QLineEdit, QLabel, QTextBrowser, QFileDialog

# from data_prep import load_data, load_images
from CNN import CNN

class Gui(QMainWindow):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

        #left panel

        #right panel
        self.data_path_button = QPushButton("Load Data")
        self.data_path_button.clicked.connect(self.open_data_window)

        #bottom panel
        #To do: make this work somehow
        self.consolePrint_left = QLineEdit()
        self.consolePrint_left.setReadOnly(True)
        self.consolePrint_left.setPlaceholderText('console print')
        self.consolePrint_left.setMinimumSize(100, 200)
        self.consoleInput_left = QLineEdit()
        self.consoleInput_left.setPlaceholderText('console input')
        self.consolePrint_right = QLineEdit()
        self.consolePrint_right.setReadOnly(True)
        self.consolePrint_right.setPlaceholderText('console print')
        self.consolePrint_right.setMinimumSize(100, 200)
        self.consoleInput_right = QLineEdit()
        self.consoleInput_right.setPlaceholderText('console input')

        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        self.setWindowTitle('inzynierka')
        self.setFixedSize(1200, 1000)

        left_panel = QGridLayout()
        left_panel.addWidget(self.data_path_button, 0, 0)

        right_panel = QGridLayout()

        top_layout = QHBoxLayout()
        top_layout.addLayout(left_panel)
        top_layout.addLayout(right_panel)

        bottom_layout = QGridLayout()
        bottom_layout.addWidget(self.consolePrint_left, 0, 0)
        bottom_layout.addWidget(self.consolePrint_right, 0, 1)
        bottom_layout.addWidget(self.consoleInput_left, 1, 0)
        bottom_layout.addWidget(self.consoleInput_right, 1, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def open_data_window(self):
        dialog = LoadDataWindow(self)
        dialog.show()


class LoadDataWindow(QMainWindow):
    def __init__(self, parent=None):
        super(LoadDataWindow, self).__init__(parent)
        self.setWindowTitle('Load Data')
        self.setFixedSize(400, 200)

        #Load Training data
        self.training_data_button = QPushButton("Load Training Data")
        self.training_Data_lineedit = QLineEdit()
        self.training_Data_lineedit.setReadOnly(True)
        self.training_data_button.clicked.connect(self.openTrainingDir)

        #Load Test data
        self.test_data_button = QPushButton("Load Test Data")
        self.test_data_lineedit = QLineEdit()
        self.test_data_lineedit.setReadOnly(True)
        self.test_data_button.clicked.connect(self.openTestDir)

        #Start fitting CNN and close window
        self.ok_button = QPushButton("Ok")

        #Uncomment for a quick CNN run
        # self.ok_button.clicked.connect(self.startCNN)

        self.ok_button.clicked.connect(self.close)

        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        main_layout = QGridLayout()

        main_layout.addWidget(self.training_Data_lineedit, 0, 0, 1, 3)
        main_layout.addWidget(self.training_data_button, 0, 4, 1, 1)
        main_layout.addWidget(self.test_data_lineedit, 1, 0, 1, 3)
        main_layout.addWidget(self.test_data_button, 1, 4, 1, 1)
        main_layout.addWidget(self.ok_button, 2, 1, 1, 2)

        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)


    #Self explanatory tbh
    def openTrainingDir(self):
        self.trainDir = QFileDialog.getExistingDirectory()
        self.training_Data_lineedit.setText(self.trainDir)

    def openTestDir(self):
        self.testDir = QFileDialog.getExistingDirectory()
        self.test_data_lineedit.setText(self.testDir)

    def startCNN(self):
        neuralNetworks = CNN(self.trainDir, self.testDir)