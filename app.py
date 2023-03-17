import cv2
import numpy as np
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# 支持不同系统的中文字体
import sys
if sys.platform == 'darwin':
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB"]
elif sys.platform == 'win32':
    plt.rcParams["font.sans-serif"] = ["SimHei"]
else:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 固定尺寸
def resizeImg(image, height=900):
    h, w = image.shape[:2]
    pro = height / h
    size = (int(w * pro), int(height))
    img = cv2.resize(image, size)
    return img


# 边缘检测
def getCanny(image):
    # 高斯滤波
    binary = cv2.GaussianBlur(image, (3, 3), 2, 2)
    # 边缘检测
    binary = cv2.Canny(binary, 60, 240, apertureSize=3)
    # 膨胀操作，尽量使边缘闭合
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return binary


# 求出面积最大的轮廓
def findMaxContour(image):
    # 寻找边缘
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 计算面积
    max_area = 0.0
    max_contour = []
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > max_area:
            max_area = currentArea
            max_contour = contour
    return max_contour, max_area


# 多边形拟合凸包的四个顶点
def getBoxPoint(contour):
    # 多边形拟合凸包
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx = approx.reshape((len(approx), 2))
    return approx


# 适配原四边形点集
def adaPoint(box, pro):
    box_pro = box
    if pro != 1.0:
        box_pro = box/pro
    box_pro = np.trunc(box_pro)
    return box_pro


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def adaptive_thres(img, win=9, beta=0.9):
    if win % 2 == 0: win = win - 1
    # 边界的均值有点麻烦
    # 这里分别计算和和邻居数再相除
    kern = np.ones([win, win])
    sums = signal.correlate2d(img, kern, 'same')
    cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
    means = sums // cnts
    # 如果直接采用均值作为阈值，背景会变花
    # 但是相邻背景颜色相差不大
    # 所以乘个系数把它们过滤掉
    img = np.where(img < means * beta, 0, 255)
    return img.astype(np.float32)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('图片边缘检测、矫正与增强锐化系统')
        self.geometry('1200x600')
        self.resizable(0, 0)
        self.make_widgets()

    def make_widgets(self):
        # 顶部菜单
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label='打开', command=self.open_file)
        self.filemenu.add_command(label='保存', command=self.save_file)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='退出', command=self.quit)
        self.menubar.add_cascade(label='文件', menu=self.filemenu)
        self.config(menu=self.menubar)
        
        self.figure_origin = Figure(figsize=(6, 6), dpi=100)
        self.figure_origin.set_facecolor('white')
        self.canvas_origin = FigureCanvasTkAgg(self.figure_origin, self)
        self.canvas_origin.get_tk_widget().grid(row=0, column=0)
        self.figure_origin.tight_layout()
        self.canvas_origin.draw()

        self.figure_result = Figure(figsize=(6, 6), dpi=100)
        self.figure_result.set_facecolor('white')
        self.canvas_result = FigureCanvasTkAgg(self.figure_result, self)
        self.canvas_result.get_tk_widget().grid(row=0, column=1)
        self.figure_result.tight_layout()
        self.canvas_result.draw()
        
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[('图片', '*.jpg *.png *.jpeg')])
        if file_path:
            self.origin = cv2.imread(file_path)
            self.result = App.enhance(self.origin)
            self.show_image()

    def show_image(self):
        self.figure_origin.clear()
        self.figure_result.clear()
        ax1 = self.figure_origin.add_subplot(111)
        ax2 = self.figure_result.add_subplot(111)
        ax1.imshow(cv2.cvtColor(self.origin, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.set_title('原图')
        ax2.set_title('结果')
        self.canvas_origin.draw()
        self.canvas_result.draw()
    
    def save_file(self):
        file_path = filedialog.asksaveasfilename(filetypes=[('图片', '*.jpg *.png *.jpeg')])
        if file_path:
            cv2.imwrite(file_path, self.result)


    @staticmethod
    def enhance(image):
        ratio = 900 / image.shape[0]
        img = resizeImg(image)
        binary_img = getCanny(img)
        max_contour, max_area = findMaxContour(binary_img)
        boxes = getBoxPoint(max_contour)
        boxes = adaPoint(boxes, ratio)
        boxes = orderPoints(boxes)
        warped = warpImage(image, boxes)
        thre = adaptive_thres(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), win=9, beta=0.9)
        return thre
        

if __name__ == '__main__':
    app = App()
    app.mainloop()
    