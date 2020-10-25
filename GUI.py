from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGridLayout, QPushButton, QHBoxLayout, QVBoxLayout, QWidget, \
    QLineEdit, QLabel, QTextBrowser


class Gui(QMainWindow):
    def __init__(self, parent=None):
        super(Gui, self).__init__(parent)

        self.data_path_button = QPushButton("Load Data")

        self.prepare_gui()
        self.show()

    def prepare_gui(self):
        self.setWindowTitle('inzynierka')
        self.setFixedSize(800, 700)

        main_layout = QHBoxLayout()
        self.main_label = QLabel()
        right_panel = QGridLayout()

        right_panel.addWidget(self.data_path_button, 0, 0)
        self.data_path_button.clicked.connect(self.open_data_window)
        main_layout.addLayout(right_panel)

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

        self.training_data_button = QPushButton("Load Training Data")
        self.training_Data_lineedit = QLineEdit()
        self.training_Data_lineedit.setReadOnly(True)

        self.test_data_button = QPushButton("Load Test Data")
        self.test_data_lineedit = QLineEdit()
        self.test_data_lineedit.setReadOnly(True)

        self.ok_button = QPushButton("Ok")

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