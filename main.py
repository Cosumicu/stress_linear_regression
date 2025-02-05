from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('final3_data.csv')

# Calculate mean of each category
df['f2f_sq_mean_row'] = df[['F2FSQQ1', 'F2FSQQ2', 'F2FSQQ3', 'F2FSQQ4', 'F2FSQQ5']].mean(axis=1)
df['f2f_ap_mean_row'] = df[['F2FAPQ1', 'F2FAPQ2', 'F2FAPQ3', 'F2FAPQ4', 'F2FAPQ5']].mean(axis=1)
df['f2f_sl_mean_row'] = df[['F2FSLQ1', 'F2FSLQ2', 'F2FSLQ3', 'F2FSLQ4', 'F2FSLQ5']].mean(axis=1)
df['f2f_en_mean_row'] = df[['F2FENQ1', 'F2FENQ2', 'F2FENQ3', 'F2FENQ4', 'F2FENQ5']].mean(axis=1)

# Calculate mean across the means of all categories and store it in a separate column
df['overall_mean_f2f'] = df[['f2f_sq_mean_row', 'f2f_ap_mean_row', 'f2f_sl_mean_row', 'f2f_en_mean_row']].mean(
    axis=1)

df['ol_sq_mean_row'] = df[['OLSQQ1', 'OLSQQ2', 'OLSQQ3', 'OLSQQ4', 'OLSQQ5']].mean(axis=1)
df['ol_ap_mean_row'] = df[['OLAPQ1', 'OLAPQ2', 'OLAPQ3', 'OLAPQ4', 'OLAPQ5']].mean(axis=1)
df['ol_sl_mean_row'] = df[['OLSLQ1', 'OLSLQ2', 'OLSLQ3', 'OLSLQ4', 'OLSLQ5']].mean(axis=1)
df['ol_en_mean_row'] = df[['OLENQ1', 'OLENQ2', 'OLENQ3', 'OLENQ4', 'OLENQ5']].mean(axis=1)

df['overall_mean_ol'] = df[['ol_sq_mean_row', 'ol_ap_mean_row', 'ol_sl_mean_row', 'ol_en_mean_row']].mean(
    axis=1)

from mpldatacursor import datacursor  # Import data cursor tool
datacursor()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1094, 602)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 201, 561))
        self.groupBox.setObjectName("groupBox")
        self.button_1 = QtWidgets.QPushButton(self.groupBox)
        self.button_1.setGeometry(QtCore.QRect(0, 0, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_1.setFont(font)
        self.button_1.setObjectName("button_1")
        self.button_2 = QtWidgets.QPushButton(self.groupBox)
        self.button_2.setGeometry(QtCore.QRect(0, 80, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_2.setFont(font)
        self.button_2.setObjectName("button_2")
        self.button_3 = QtWidgets.QPushButton(self.groupBox)
        self.button_3.setGeometry(QtCore.QRect(0, 160, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_3.setFont(font)
        self.button_3.setObjectName("button_3")
        self.button_4 = QtWidgets.QPushButton(self.groupBox)
        self.button_4.setGeometry(QtCore.QRect(0, 240, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_4.setFont(font)
        self.button_4.setObjectName("button_4")
        self.button_5 = QtWidgets.QPushButton(self.groupBox)
        self.button_5.setGeometry(QtCore.QRect(0, 360, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_5.setFont(font)
        self.button_5.setObjectName("button_5")
        self.button_6 = QtWidgets.QPushButton(self.groupBox)
        self.button_6.setGeometry(QtCore.QRect(0, 440, 200, 80))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.button_6.setFont(font)
        self.button_6.setObjectName("button_6")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(6, 320, 191, 41))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(200, 0, 891, 571))
        self.stackedWidget.setObjectName("stackedWidget")
        self.p1 = QtWidgets.QWidget()
        self.p1.setObjectName("p1")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.p1)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 30, 891, 411))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stackedWidget.addWidget(self.p1)
        self.p2 = QtWidgets.QWidget()
        self.p2.setObjectName("p2")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.p2)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 30, 891, 411))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.stackedWidget.addWidget(self.p2)
        self.p3 = QtWidgets.QWidget()
        self.p3.setObjectName("p3")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.p3)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 30, 891, 411))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.stackedWidget.addWidget(self.p3)
        self.p4 = QtWidgets.QWidget()
        self.p4.setObjectName("p4")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.p4)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(0, 30, 891, 411))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.stackedWidget.addWidget(self.p4)
        self.p5 = QtWidgets.QWidget()
        self.p5.setObjectName("p5")
        self.gridLayoutWidget = QtWidgets.QWidget(self.p5)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1000, 800))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.stackedWidget.addWidget(self.p5)
        self.p6 = QtWidgets.QWidget()
        self.p6.setObjectName("p6")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.p6)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 1000, 800))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.stackedWidget.addWidget(self.p6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1094, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # BUFFS
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - MainWindow.width()) // 2
        y = (screen_geometry.height() - MainWindow.height()) // 2
        MainWindow.move(x, y)

        # CALL FUNCTIONS
        self.stacked_widget_fun()

        # SHOWING GRAPHS
        self.canvas1 = FigureCanvas(Figure())
        self.horizontalLayout.addWidget(self.canvas1)
        self.ax1_1 = self.canvas1.figure.add_subplot(111)

        self.canvas2 = FigureCanvas(Figure())
        self.horizontalLayout.addWidget(self.canvas2)
        self.ax1_2 = self.canvas2.figure.add_subplot(111)

        # GENDER
        self.canvas3 = FigureCanvas(Figure())
        self.horizontalLayout_2.addWidget(self.canvas3)
        self.ax3_1 = self.canvas3.figure.add_subplot(111)

        self.canvas4 = FigureCanvas(Figure())
        self.horizontalLayout_2.addWidget(self.canvas4)
        self.ax3_2 = self.canvas4.figure.add_subplot(111)

        self.canvas5 = FigureCanvas(Figure())
        self.horizontalLayout_3.addWidget(self.canvas5)
        self.ax4_1 = self.canvas5.figure.add_subplot(111)

        self.canvas6 = FigureCanvas(Figure())
        self.horizontalLayout_3.addWidget(self.canvas6)
        self.ax4_2 = self.canvas6.figure.add_subplot(111)

        self.canvas7 = FigureCanvas(Figure())
        self.horizontalLayout_4.addWidget(self.canvas7)
        self.ax5_1 = self.canvas7.figure.add_subplot(111)

        self.canvas8 = FigureCanvas(Figure())
        self.horizontalLayout_4.addWidget(self.canvas8)
        self.ax5_2 = self.canvas8.figure.add_subplot(111)

        # REGRESSION
        self.canvas9 = FigureCanvas(Figure())
        self.gridLayout.addWidget(self.canvas9, 0, 0)
        self.ax6_1 = self.canvas9.figure.add_subplot(111)

        self.canvas10 = FigureCanvas(Figure())
        self.gridLayout.addWidget(self.canvas10, 0, 1)
        self.ax6_2 = self.canvas10.figure.add_subplot(111)

        self.canvas11 = FigureCanvas(Figure())
        self.gridLayout.addWidget(self.canvas11, 1, 0)
        self.ax6_3 = self.canvas11.figure.add_subplot(111)

        self.canvas12 = FigureCanvas(Figure())
        self.gridLayout.addWidget(self.canvas12, 1, 1)
        self.ax6_4 = self.canvas12.figure.add_subplot(111)

        self.canvas13 = FigureCanvas(Figure())
        self.gridLayout_2.addWidget(self.canvas13, 0, 0)
        self.ax7_1 = self.canvas13.figure.add_subplot(111)

        self.canvas14 = FigureCanvas(Figure())
        self.gridLayout_2.addWidget(self.canvas14, 0, 1)
        self.ax7_2 = self.canvas14.figure.add_subplot(111)

        self.canvas15 = FigureCanvas(Figure())
        self.gridLayout_2.addWidget(self.canvas15, 1, 0)
        self.ax7_3 = self.canvas15.figure.add_subplot(111)

        self.canvas16 = FigureCanvas(Figure())
        self.gridLayout_2.addWidget(self.canvas16, 1, 1)
        self.ax7_4 = self.canvas16.figure.add_subplot(111)


        # CALL FUNCTION (GRAPHS)
        self.gathered_data_f2f_graph()
        self.gathered_data_ol_graph()

        self.gathered_data_gender_f2f_graph()
        self.gathered_data_gender_ol_graph()

        self.gathered_data_program_f2f_graph()
        self.gathered_data_program_ol_graph()

        self.gathered_data_level_f2f_graph()
        self.gathered_data_level_ol_graph()

        # LINEAR REGRESSION
        print("F2F")
        self.regression_f2f_sq()
        self.regression_f2f_ap()
        self.regression_f2f_sl()
        self.regression_f2f_en()

        print("ONLINE")
        self.regression_ol_sq()
        self.regression_ol_ap()
        self.regression_ol_sl()
        self.regression_ol_en()

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DATA REPRESENTATION"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.button_1.setText(_translate("MainWindow", "GATHERED DATA"))
        self.button_2.setText(_translate("MainWindow", "GATHERED DATA (GENDER)"))
        self.button_3.setText(_translate("MainWindow", "GATHERED DATA (PROGRAM)"))
        self.button_4.setText(_translate("MainWindow", "GATHERED DATA (YEAR LEVEL)"))
        self.button_5.setText(_translate("MainWindow", "FACE TO FACE"))
        self.button_6.setText(_translate("MainWindow", "ONLINE"))
        self.label.setText(_translate("MainWindow", "PREDICTIVE MODEL"))

    def stacked_widget_fun(self):
        self.button_1.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p1))
        self.button_2.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p2))
        self.button_3.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p3))
        self.button_4.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p4))
        self.button_5.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p5))
        self.button_6.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.p6))

        self.button_1.clicked.connect(self.resizeStackedWidgetPage)
        self.button_2.clicked.connect(self.resizeStackedWidgetPage)
        self.button_3.clicked.connect(self.resizeStackedWidgetPage)
        self.button_4.clicked.connect(self.resizeStackedWidgetPage)
        self.button_5.clicked.connect(self.resizeStackedWidgetPage)
        self.button_6.clicked.connect(self.resizeStackedWidgetPage)


    def resizeStackedWidgetPage(self):
        if self.stackedWidget.currentWidget() in [self.p5, self.p6]:
            self.stackedWidget.setGeometry(QtCore.QRect(200, 0, 1091, 801))
            MainWindow.resize(1200, 832)
            screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
            x = (screen_geometry.width() - MainWindow.width()) // 2
            y = (screen_geometry.height() - MainWindow.height()) // 2
            MainWindow.move(x, y)
        else:
            # Set the geometry and resize to original size
            self.stackedWidget.setGeometry(QtCore.QRect(200, 0, 891, 571))
            MainWindow.resize(1094, 602)
            screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
            x = (screen_geometry.width() - MainWindow.width()) // 2
            y = (screen_geometry.height() - MainWindow.height()) // 2
            MainWindow.move(x, y)

    def gathered_data_f2f_graph(self):
        f2f_sq_mean = df[['F2FSQQ1', 'F2FSQQ2', 'F2FSQQ3', 'F2FSQQ4', 'F2FSQQ5']].mean().mean()
        f2f_ap_mean = df[['F2FAPQ1', 'F2FAPQ2', 'F2FAPQ3', 'F2FAPQ4', 'F2FAPQ5']].mean().mean()
        f2f_sl_mean = df[['F2FSLQ1', 'F2FSLQ2', 'F2FSLQ3', 'F2FSLQ4', 'F2FSLQ5']].mean().mean()
        f2f_en_mean = df[['F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1']].mean().mean()

        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y = [f2f_sq_mean, f2f_ap_mean, f2f_sl_mean, f2f_en_mean]
        self.ax1_1.bar(x, y, color = '#D2B4DE')

        self.ax1_1.set_xlabel('Factors')
        self.ax1_1.set_ylabel('Stress Level')
        self.ax1_1.set_title('F2F Summary')
        self.ax1_1.set_ylim(0, 5)

        for bars in [self.ax1_1.patches[i:i + len(x)] for i in range(0, len(self.ax1_1.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax1_1.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        self.canvas1.draw()

    def gathered_data_ol_graph(self):
        ol_sq_mean = df[['OLSQQ1', 'OLSQQ2', 'OLSQQ3', 'OLSQQ4', 'OLSQQ5']].mean().mean()
        ol_ap_mean = df[['OLAPQ1', 'OLAPQ2', 'OLAPQ3', 'OLAPQ4', 'OLAPQ5']].mean().mean()
        ol_sl_mean = df[['OLSLQ1', 'OLSLQ2', 'OLSLQ3', 'OLSLQ4', 'OLSLQ5']].mean().mean()
        ol_en_mean = df[['OLENQ1', 'OLENQ2', 'OLENQ3', 'OLENQ4', 'OLENQ5']].mean().mean()

        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y = [ol_sq_mean, ol_ap_mean, ol_sl_mean, ol_en_mean]
        self.ax1_2.bar(x, y, color = '#D2B4DE')

        self.ax1_2.set_xlabel('Factors')
        self.ax1_2.set_ylabel('Stress Level')
        self.ax1_2.set_title('ONLINE Summary')
        self.ax1_2.set_ylim(0, 5)

        for bars in [self.ax1_2.patches[i:i + len(x)] for i in range(0, len(self.ax1_2.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax1_2.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        self.canvas2.draw()

    def gathered_data_gender_f2f_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Gender')

        # Calculate means for each gender group separately
        f2f_sq_mean = grouped_df[['F2FSQQ1', 'F2FSQQ2', 'F2FSQQ3', 'F2FSQQ4', 'F2FSQQ5']].mean()
        f2f_ap_mean = grouped_df[['F2FAPQ1', 'F2FAPQ2', 'F2FAPQ3', 'F2FAPQ4', 'F2FAPQ5']].mean()
        f2f_sl_mean = grouped_df[['F2FSLQ1', 'F2FSLQ2', 'F2FSLQ3', 'F2FSLQ4', 'F2FSLQ5']].mean()
        f2f_en_mean = grouped_df[['F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_male = [f2f_sq_mean.loc['Male'].mean(), f2f_ap_mean.loc['Male'].mean(), f2f_sl_mean.loc['Male'].mean(),
                  f2f_en_mean.loc['Male'].mean()]
        y_female = [f2f_sq_mean.loc['Female'].mean(), f2f_ap_mean.loc['Female'].mean(),
                    f2f_sl_mean.loc['Female'].mean(), f2f_en_mean.loc['Female'].mean()]

        # Plotting
        bar_width = 0.35  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax3_1.bar(x_indexes - bar_width / 2, y_male, bar_width, label='Male', color='#ADD8E6')
        self.ax3_1.bar(x_indexes + bar_width / 2, y_female, bar_width, label='Female', color='#FFB6C1')

        # Setting labels and title
        self.ax3_1.set_xticks(x_indexes)
        self.ax3_1.set_xticklabels(x)
        self.ax3_1.set_xlabel('Factors')
        self.ax3_1.set_ylabel('Stress Level')
        self.ax3_1.set_title('F2F Summary')
        self.ax3_1.legend()
        self.ax3_1.set_ylim(0, 5)

        for bars in [self.ax3_1.patches[i:i + len(x)] for i in range(0, len(self.ax3_1.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax3_1.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        # Drawing canvas
        self.canvas3.draw()

    def gathered_data_gender_ol_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Gender')

        # Calculate means for each gender group separately
        ol_sq_mean = grouped_df[['OLSQQ1', 'OLSQQ2', 'OLSQQ3', 'OLSQQ4', 'OLSQQ5']].mean()
        ol_ap_mean = grouped_df[['OLAPQ1', 'OLAPQ2', 'OLAPQ3', 'OLAPQ4', 'OLAPQ5']].mean()
        ol_sl_mean = grouped_df[['OLSLQ1', 'OLSLQ2', 'OLSLQ3', 'OLSLQ4', 'OLSLQ5']].mean()
        ol_en_mean = grouped_df[['OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_male = [ol_sq_mean.loc['Male'].mean(), ol_ap_mean.loc['Male'].mean(), ol_sl_mean.loc['Male'].mean(),
                  ol_en_mean.loc['Male'].mean()]
        y_female = [ol_sq_mean.loc['Female'].mean(), ol_ap_mean.loc['Female'].mean(),
                    ol_sl_mean.loc['Female'].mean(), ol_en_mean.loc['Female'].mean()]

        # Plotting
        bar_width = 0.35  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax3_2.bar(x_indexes - bar_width / 2, y_male, bar_width, label='Male', color='#ADD8E6')
        self.ax3_2.bar(x_indexes + bar_width / 2, y_female, bar_width, label='Female', color='#FFB6C1')

        # Setting labels and title
        self.ax3_2.set_xticks(x_indexes)
        self.ax3_2.set_xticklabels(x)
        self.ax3_2.set_xlabel('Factors')
        self.ax3_2.set_ylabel('Stress Level')
        self.ax3_2.set_title('ONLINE Summary')
        self.ax3_2.legend()
        self.ax3_2.set_ylim(0, 5)

        for bars in [self.ax3_2.patches[i:i + len(x)] for i in range(0, len(self.ax3_2.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax3_2.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        # Drawing canvas
        self.canvas4.draw()

    def gathered_data_program_f2f_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Program')

        # Calculate means for each gender group separately
        f2f_sq_mean = grouped_df[['F2FSQQ1', 'F2FSQQ2', 'F2FSQQ3', 'F2FSQQ4', 'F2FSQQ5']].mean()
        f2f_ap_mean = grouped_df[['F2FAPQ1', 'F2FAPQ2', 'F2FAPQ3', 'F2FAPQ4', 'F2FAPQ5']].mean()
        f2f_sl_mean = grouped_df[['F2FSLQ1', 'F2FSLQ2', 'F2FSLQ3', 'F2FSLQ4', 'F2FSLQ5']].mean()
        f2f_en_mean = grouped_df[['F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_cs = [f2f_sq_mean.loc['BSCS'].mean(), f2f_ap_mean.loc['BSCS'].mean(), f2f_sl_mean.loc['BSCS'].mean(),
                  f2f_en_mean.loc['BSCS'].mean()]
        y_it = [f2f_sq_mean.loc['BSIT'].mean(), f2f_ap_mean.loc['BSIT'].mean(),
                    f2f_sl_mean.loc['BSIT'].mean(), f2f_en_mean.loc['BSIT'].mean()]

        # Plotting
        bar_width = 0.35  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax4_1.bar(x_indexes - bar_width / 2, y_cs, bar_width, label='BSCS', color='#F9E79F')
        self.ax4_1.bar(x_indexes + bar_width / 2, y_it, bar_width, label='BSIT', color='#AED6F1')

        # Setting labels and title
        self.ax4_1.set_xticks(x_indexes)
        self.ax4_1.set_xticklabels(x)
        self.ax4_1.set_xlabel('Factors')
        self.ax4_1.set_ylabel('Stress Level')
        self.ax4_1.set_title('F2F Summary')
        self.ax4_1.legend()
        self.ax4_1.set_ylim(0, 5)

        for bars in [self.ax4_1.patches[i:i + len(x)] for i in range(0, len(self.ax4_1.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax4_1.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        # Drawing canvas
        self.canvas5.draw()

    def gathered_data_program_ol_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Program')

        # Calculate means for each gender group separately
        ol_sq_mean = grouped_df[['OLSQQ1', 'OLSQQ2', 'OLSQQ3', 'OLSQQ4', 'OLSQQ5']].mean()
        ol_ap_mean = grouped_df[['OLAPQ1', 'OLAPQ2', 'OLAPQ3', 'OLAPQ4', 'OLAPQ5']].mean()
        ol_sl_mean = grouped_df[['OLSLQ1', 'OLSLQ2', 'OLSLQ3', 'OLSLQ4', 'OLSLQ5']].mean()
        ol_en_mean = grouped_df[['OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_cs = [ol_sq_mean.loc['BSCS'].mean(), ol_ap_mean.loc['BSCS'].mean(), ol_sl_mean.loc['BSCS'].mean(),
                  ol_en_mean.loc['BSCS'].mean()]
        y_it = [ol_sq_mean.loc['BSIT'].mean(), ol_ap_mean.loc['BSIT'].mean(),
                    ol_sl_mean.loc['BSIT'].mean(), ol_en_mean.loc['BSIT'].mean()]

        # Plotting
        bar_width = 0.35  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax4_2.bar(x_indexes - bar_width / 2, y_cs, bar_width, label='BSCS', color='#F9E79F')
        self.ax4_2.bar(x_indexes + bar_width / 2, y_it, bar_width, label='BSIT', color='#AED6F1')

        # Setting labels and title
        self.ax4_2.set_xticks(x_indexes)
        self.ax4_2.set_xticklabels(x)
        self.ax4_2.set_xlabel('Factors')
        self.ax4_2.set_ylabel('Stress Level')
        self.ax4_2.set_title('OL Summary')
        self.ax4_2.legend()
        self.ax4_2.set_ylim(0, 5)

        for bars in [self.ax4_2.patches[i:i + len(x)] for i in range(0, len(self.ax4_2.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax4_2.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)

        # Drawing canvas
        self.canvas6.draw()

    def gathered_data_level_f2f_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Year Level')

        # Calculate means for each gender group separately
        f2f_sq_mean = grouped_df[['F2FSQQ1', 'F2FSQQ2', 'F2FSQQ3', 'F2FSQQ4', 'F2FSQQ5']].mean()
        f2f_ap_mean = grouped_df[['F2FAPQ1', 'F2FAPQ2', 'F2FAPQ3', 'F2FAPQ4', 'F2FAPQ5']].mean()
        f2f_sl_mean = grouped_df[['F2FSLQ1', 'F2FSLQ2', 'F2FSLQ3', 'F2FSLQ4', 'F2FSLQ5']].mean()
        f2f_en_mean = grouped_df[['F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1', 'F2FENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_1 = [f2f_sq_mean.loc['First Year'].mean(), f2f_ap_mean.loc['First Year'].mean(), f2f_sl_mean.loc['First Year'].mean(),
                  f2f_en_mean.loc['First Year'].mean()]
        y_2 = [f2f_sq_mean.loc['Second Year'].mean(), f2f_ap_mean.loc['Second Year'].mean(),
                    f2f_sl_mean.loc['Second Year'].mean(), f2f_en_mean.loc['Second Year'].mean()]
        y_3 = [f2f_sq_mean.loc['Third Year'].mean(), f2f_ap_mean.loc['Third Year'].mean(),
               f2f_sl_mean.loc['Third Year'].mean(), f2f_en_mean.loc['Third Year'].mean()]
        y_4 = [f2f_sq_mean.loc['Fourth Year'].mean(), f2f_ap_mean.loc['Fourth Year'].mean(),
               f2f_sl_mean.loc['Fourth Year'].mean(), f2f_en_mean.loc['Fourth Year'].mean()]

        # Plotting
        bar_width = 0.2  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax5_1.bar(x_indexes - bar_width, y_1, bar_width, label='1st Year', color='#ABEBC6')
        self.ax5_1.bar(x_indexes, y_2, bar_width, label='2nd Year', color='#AED6F1')
        self.ax5_1.bar(x_indexes + bar_width, y_3, bar_width, label='3rd Year', color='#D2B4DE')
        self.ax5_1.bar(x_indexes + 2 * bar_width, y_4, bar_width, label='4th Year', color='#F9E79F')

        # Setting labels and title
        self.ax5_1.set_xticks(x_indexes + bar_width / 2)
        self.ax5_1.set_xticklabels(x)
        self.ax5_1.set_xlabel('Factors')
        self.ax5_1.set_ylabel('Stress Level')
        self.ax5_1.set_title('F2F Summary by Program')
        self.ax5_1.set_ylim(0, 5)

        for bars in [self.ax5_1.patches[i:i + len(x)] for i in range(0, len(self.ax5_1.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax5_1.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=5)

        # Drawing canvas
        self.canvas7.draw()

    def gathered_data_level_ol_graph(self):
        # Grouping by gender
        grouped_df = df.groupby('Year Level')

        # Calculate means for each gender group separately
        ol_sq_mean = grouped_df[['OLSQQ1', 'OLSQQ2', 'OLSQQ3', 'OLSQQ4', 'OLSQQ5']].mean()
        ol_ap_mean = grouped_df[['OLAPQ1', 'OLAPQ2', 'OLAPQ3', 'OLAPQ4', 'OLAPQ5']].mean()
        ol_sl_mean = grouped_df[['OLSLQ1', 'OLSLQ2', 'OLSLQ3', 'OLSLQ4', 'OLSLQ5']].mean()
        ol_en_mean = grouped_df[['OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1', 'OLENQ1']].mean()

        # Extracting means for plotting
        x = ['Sleep', 'Performance', 'Social', 'Environment']
        y_1 = [ol_sq_mean.loc['First Year'].mean(), ol_ap_mean.loc['First Year'].mean(), ol_sl_mean.loc['First Year'].mean(),
                  ol_en_mean.loc['First Year'].mean()]
        y_2 = [ol_sq_mean.loc['Second Year'].mean(), ol_ap_mean.loc['Second Year'].mean(),
                    ol_sl_mean.loc['Second Year'].mean(), ol_en_mean.loc['Second Year'].mean()]
        y_3 = [ol_sq_mean.loc['Third Year'].mean(), ol_ap_mean.loc['Third Year'].mean(),
               ol_sl_mean.loc['Third Year'].mean(), ol_en_mean.loc['Third Year'].mean()]
        y_4 = [ol_sq_mean.loc['Fourth Year'].mean(), ol_ap_mean.loc['Fourth Year'].mean(),
               ol_sl_mean.loc['Fourth Year'].mean(), ol_en_mean.loc['Fourth Year'].mean()]

        # Plotting
        bar_width = 0.2  # Width of the bars
        x_indexes = np.arange(len(x))  # Indexes for the x-axis values

        self.ax5_2.bar(x_indexes - bar_width, y_1, bar_width, label='1st Year', color='#ABEBC6')
        self.ax5_2.bar(x_indexes, y_2, bar_width, label='2nd Year', color='#AED6F1')
        self.ax5_2.bar(x_indexes + bar_width, y_3, bar_width, label='3rd Year', color='#D2B4DE')
        self.ax5_2.bar(x_indexes + 2 * bar_width, y_4, bar_width, label='4th Year', color='#F9E79F')

        # Setting labels and title
        self.ax5_2.set_xticks(x_indexes + bar_width / 2)
        self.ax5_2.set_xticklabels(x)
        self.ax5_2.set_xlabel('Factors')
        self.ax5_2.set_ylabel('Stress Level')
        self.ax5_2.set_title('ONLINE Summary by Program')
        self.ax5_2.legend()
        self.ax5_2.set_ylim(0, 5)

        for bars in [self.ax5_2.patches[i:i + len(x)] for i in range(0, len(self.ax5_2.patches), len(x))]:
            for bar in bars:
                height = bar.get_height()
                self.ax5_2.annotate('{}'.format(round(height, 2)),
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=5)

        # Drawing canvas
        self.canvas8.draw()

    def regression_f2f_sq(self):
        #Independent variable
        X = df[['f2f_sq_mean_row']]
        # Dependent variable
        y = df['overall_mean_f2f']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Sleep Quality):", r_squared)

        # Plot the training data
        self.ax6_1 = self.canvas9.figure.add_subplot(111)
        self.ax6_1.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax6_1.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax6_1.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax6_1.set_xlabel('Sleep Quality', labelpad=20)
        self.ax6_1.set_ylabel('Stress Level (F2F)', labelpad=25)
        self.ax6_1.set_title('Sleep Quality')
        self.ax6_1.legend()

        self.ax6_1.set_xticks([])
        self.ax6_1.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas9.draw()

    def regression_f2f_ap(self):
        # Independent variable
        X = df[['f2f_ap_mean_row']]
        # Dependent variable
        y = df['overall_mean_f2f']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Academic Performance):", r_squared)

        # Plot the training data
        self.ax6_2 = self.canvas10.figure.add_subplot(111)
        self.ax6_2.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax6_2.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax6_2.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax6_2.set_xlabel('Academic Performance', labelpad=20)
        self.ax6_2.set_ylabel('Stress Level (F2F)', labelpad=25)
        self.ax6_2.set_title('Academic Performance')
        self.ax6_2.legend()

        self.ax6_2.set_xticks([])
        self.ax6_2.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas10.draw()

    def regression_f2f_sl(self):
        # Independent variable
        X = df[['f2f_sl_mean_row']]
        # Dependent variable
        y = df['overall_mean_f2f']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Social Life):", r_squared)

        # Plot the training data
        self.ax6_3 = self.canvas11.figure.add_subplot(111)
        self.ax6_3.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax6_3.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax6_3.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax6_3.set_xlabel('Social Life', labelpad= 20)
        self.ax6_3.set_ylabel('Stress Level (F2F)', labelpad= 25)
        self.ax6_3.set_title('Social Life')
        self.ax6_3.legend()

        self.ax6_3.set_xticks([])
        self.ax6_3.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas11.draw()

    def regression_f2f_en(self):
        # Independent variable
        X = df[['f2f_en_mean_row']]
        # Dependent variable
        y = df['overall_mean_f2f']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Environment):", r_squared)
        print()

        # Plot the training data
        self.ax6_4 = self.canvas12.figure.add_subplot(111)
        self.ax6_4.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax6_4.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax6_4.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax6_4.set_xlabel('Environment', labelpad = 20)
        self.ax6_4.set_ylabel('Stress Level (F2F)', labelpad = 25)
        self.ax6_4.set_title('Environment')

        self.ax6_4.set_xticks([])
        self.ax6_4.set_yticks([])

        self.ax6_4.legend()
        # Refresh the canvas to update the plot
        self.canvas12.draw()


    def regression_ol_sq(self):
        #Independent variable
        X = df[['ol_sq_mean_row']]
        # Dependent variable
        y = df['overall_mean_ol']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Sleep Quality):", r_squared)

        # Plot the training data
        self.ax7_1 = self.canvas13.figure.add_subplot(111)
        self.ax7_1.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax7_1.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax7_1.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax7_1.set_xlabel('Sleep Quality', labelpad=20)
        self.ax7_1.set_ylabel('Stress Level (ONLINE)', labelpad=25)
        self.ax7_1.set_title('Sleep Quality')
        self.ax7_1.legend()

        self.ax7_1.set_xticks([])
        self.ax7_1.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas13.draw()

    def regression_ol_ap(self):
        # Independent variable
        X = df[['ol_ap_mean_row']]
        # Dependent variable
        y = df['overall_mean_ol']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Academic Performance):", r_squared)

        # Plot the training data
        self.ax7_2 = self.canvas14.figure.add_subplot(111)
        self.ax7_2.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax7_2.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax7_2.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax7_2.set_xlabel('Academic Performance', labelpad=20)
        self.ax7_2.set_ylabel('Stress Level (ONLINE)', labelpad=25)
        self.ax7_2.set_title('Academic Performance')
        self.ax7_2.legend()

        self.ax7_2.set_xticks([])
        self.ax7_2.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas14.draw()

    def regression_ol_sl(self):
        # Independent variable
        X = df[['ol_sl_mean_row']]
        # Dependent variable
        y = df['overall_mean_ol']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Social Life):", r_squared)

        # Plot the training data
        self.ax7_3 = self.canvas15.figure.add_subplot(111)
        self.ax7_3.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax7_3.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax7_3.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax7_3.set_xlabel('Social Life', labelpad= 20)
        self.ax7_3.set_ylabel('Stress Level (ONLINE)', labelpad= 25)
        self.ax7_3.set_title('Social Life')
        self.ax7_3.legend()

        self.ax7_3.set_xticks([])
        self.ax7_3.set_yticks([])
        # Refresh the canvas to update the plot
        self.canvas15.draw()

    def regression_ol_en(self):
        # Independent variable
        X = df[['ol_en_mean_row']]
        # Dependent variable
        y = df['overall_mean_ol']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the Linear Regression model
        model = LinearRegression()
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the testing data
        y_pred = model.predict(X_test)

        # get R-squared value
        r_squared = model.score(X_test, y_test)

        print("R-squared (Environment):", r_squared)

        # Plot the training data
        self.ax7_4 = self.canvas16.figure.add_subplot(111)
        self.ax7_4.scatter(X_train, y_train, color='blue', label='Training Data')
        # Plot the testing data
        self.ax7_4.scatter(X_test, y_test, color='red', label='Testing Data')
        # Plot the regression line
        self.ax7_4.plot(X_test, y_pred, color='black', linewidth=3, label='Regression Line')
        # Add labels and title
        self.ax7_4.set_xlabel('Environment', labelpad = 20)
        self.ax7_4.set_ylabel('Stress Level (ONLINE)', labelpad = 25)
        self.ax7_4.set_title('Environment')

        self.ax7_4.set_xticks([])
        self.ax7_4.set_yticks([])

        self.ax7_4.legend()
        # Refresh the canvas to update the plot
        self.canvas16.draw()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
