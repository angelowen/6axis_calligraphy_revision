import sys,os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore,QtGui,QtWidgets
from project import demo_main, model_env
from calligraphy.code import draw_pic
from demo_utils import argument_setting

windows = []
meetingUrl="https://meet.google.com/ose-krmk-zzg"
noise=0
word_idx=42
# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         QtWidgets.QMainWindow.setFixedSize(self,1600,400)
#         self.webview = WebEngineView()
#         self.webview.load(QtCore.QUrl(meetingUrl))
#         self.setCentralWidget(self.webview)
# class WebEngineView(QtWebEngineWidgets.QWebEngineView):
#     windows = [] #创建一个容器存储每个窗口，不然会崩溃，因为是createwindow函数里面的临时变量
#     def createWindow(self, QWebEnginePage_WebWindowType):
#         newtab =   WebEngineView()
#         newwindow= MainWindow()
#         newwindow.setCentralWidget(newtab)
#         newwindow.show()
#         self.windows.append(newwindow)
#         return newtab
        
class ExComboBox(object):
    def __init__(self,w):
        print(w.comboBox.currentText())
        w.comboBox.activated[str].connect(self.words)
    def words(self):
        global word_idx
        w.input.setPixmap(QPixmap("./imgs/loading.gif"))
        w.slim.setPixmap(QPixmap("./imgs/loading.gif"))
        print(w.comboBox.currentText())
        if w.comboBox.currentText() == "永":
            w.target.setPixmap(QPixmap("imgs/YONG.jpg"))
            word_idx = 42
        elif w.comboBox.currentText() == "史":
            w.target.setPixmap(QPixmap("imgs/SHI.jpg"))
            # w.slim.setPixmap(QPixmap("output/test_char/test_all_compare.png"))
            word_idx = 436
        elif w.comboBox.currentText() == "殺":
            w.target.setPixmap(QPixmap("imgs/SHA.jpg"))
            word_idx = 312
        elif w.comboBox.currentText() == "并":
            w.target.setPixmap(QPixmap("imgs/BING.jpg"))
            word_idx = 773
        elif w.comboBox.currentText() == "引":
            w.target.setPixmap(QPixmap("imgs/YIN.jpg"))
            word_idx = 277
 
def go_web(args):
    
    noise = w.doubleSpinBox.value()
    demo_main(args, noise, word_idx)
    draw_pic()
    w.slim.setPixmap(QPixmap("output/test_char/test_all_compare.png"))
    w.input.setPixmap(QPixmap("./output/visual/test_all_input.png"))
    # newtab =   WebEngineView()
    # newtab.load(QtCore.QUrl(meetingUrl))
    # newwindow= MainWindow()
    # newwindow.setCentralWidget(newtab)
    # newwindow.show()
    # windows.append(newwindow)
    

if __name__ == "__main__":
    args = argument_setting()

    # construction env first
    if not args.nonefficient:
        args.model, args.critetion, args.extractor = model_env(args)

    if args.gui:
        app = QApplication(sys.argv)
        w = loadUi('demo.ui')
        windows.append(w)
        ui=ExComboBox(w)
        w.label.setScaledContents(True)
        w.pushButton.clicked.connect(lambda: go_web(args))
        # w.showFullScreen()
        # w.setFixedSize(800,800)
        w.show()
        app.exec_()
    else:
        demo_main(args)