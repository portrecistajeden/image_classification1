import sys
from PyQt5.QtWidgets import QApplication
from GUI import Gui

if __name__ =='__main__':
    app = QApplication(sys.argv)
    window = Gui()
    sys.exit(app.exec_())
