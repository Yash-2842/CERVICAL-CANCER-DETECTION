from tkinter import *
from tkinter import filedialog as fd
from tensorflow import keras

import numpy as np
from PIL import ImageTk,Image
import cv2

window=Tk()
window.title("CERVICAL CANCER DETECTION")

window.geometry("950x650")
window.config(bg="#0099CC")

canvas_inp = Canvas(height=190,width=190,bg="white", highlightthickness=0)
canvas_inp.place(x=280,y=60)

canvas_gray = Canvas(height=190,width=190,bg="white",highlightthickness=0)
canvas_gray.place(x=510,y=60)

canvas_hist = Canvas(height=190,width=190,bg="white",highlightthickness=0)
canvas_hist.place(x=730,y=60)

canvas_color = Canvas(height=190,width=190,bg="white",highlightthickness=0)
canvas_color.place(x=370,y=320)

canvas_edge = Canvas(height=190,width=190,bg="white",highlightthickness=0)
canvas_edge.place(x=610,y=320)

output_lab = Label(text="", font=("Times New Roman", 15, "italic"), fg="black", bg="#FF9900")

def close():
    window.quit()

resized_img = Image.open("D:/PycharmProjects/Cervical Cancer Detection/Cervical Cancer Dataset/Test/moderate_dysplastic/148848523-148848538-001 - Copy.BMP")
Photoimage_org = resized_img
path = ""

def browse():
    global resized_img,Photoimage_org,path,output_lab
    path = fd.askopenfile()
    img = cv2.imread(path.name)
    resized_img = cv2.resize(img,(190, 190))
    Photoimage_org = Image.fromarray(resized_img)
    Photoimage_org = ImageTk.PhotoImage(Photoimage_org)
    canvas_inp.create_image(0, 0, anchor=NW, image=Photoimage_org)
    canvas_gray.delete("all")
    canvas_hist.delete("all")
    canvas_color.delete("all")
    canvas_edge.delete("all")
    output_lab.destroy()
    print(path.name)

PhotoImage_gray = resized_img
gray_image = resized_img
def gray():
    global PhotoImage_gray,resized_img,gray_image
    gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    PhotoImage_gray = Image.fromarray(gray_image)
    PhotoImage_gray = ImageTk.PhotoImage(PhotoImage_gray)
    canvas_gray.create_image(0,0,anchor=NW, image = PhotoImage_gray)

PhotoImage_hist = resized_img
hist_image = resized_img
def histogram():
    global PhotoImage_hist,hist_image,gray_image
    hist_image = cv2.equalizeHist(gray_image)
    PhotoImage_hist = Image.fromarray(hist_image)
    PhotoImage_hist = ImageTk.PhotoImage(PhotoImage_hist)
    canvas_hist.create_image(0, 0, anchor=NW, image=PhotoImage_hist)

PhotoImage_color = resized_img
color_img = resized_img
def coloredHist():
    global PhotoImage_color,hist_image,color_img
    color_img = cv2.cvtColor(hist_image,cv2.COLOR_GRAY2BGR)
    PhotoImage_color = Image.fromarray(color_img)
    PhotoImage_color = ImageTk.PhotoImage(PhotoImage_color)
    canvas_color.create_image(0, 0, anchor=NW, image=PhotoImage_color)

PhotoImage_edge = resized_img
def EdgeDetect():
    global PhotoImage_edge,color_img,gray_image
    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]
    edge_img = cv2.Canny(im_bw,0,70)
    contours, hierarchy = cv2.findContours(edge_img.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(color_img, contours, -1, (0, 255, 0), 3)
    PhotoImage_edge = Image.fromarray(color_img)
    PhotoImage_edge = ImageTk.PhotoImage(PhotoImage_edge)
    canvas_edge.create_image(0,0,anchor=NW, image=PhotoImage_edge)

def CancerDetect():
    global path,output_lab
    classes = {0: "normal_columnar", 1: "normal_intermediate", 2: "normal_superficiel", 3: "light_dysplastic",
               4: "moderate_dysplastic", 5: "severe_dysplastic", 6: "carcinoma_in_situ"}
    model = keras.models.load_model('D:\PycharmProjects\Cervical Cancer Detection\cervical_model.h5')

    img = cv2.imread(path.name)
    resize_img = cv2.resize(img,(50,50))
    gray = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    color = cv2.cvtColor(hist,cv2.COLOR_GRAY2BGR)
    resize_img = color.reshape(1,50,50,3)
    resize_img = np.array(resize_img)/255.0
    testnp = model.predict(resize_img)
    print(testnp)
    maxy = np.max(testnp.reshape(-1, ))
    out = np.where(testnp.reshape(-1, ) == maxy)
    print(classes[out[0][0]])

    output_lab = Label(text="", font=("Times New Roman", 15, "italic"), fg="black", bg="#FF9900")
    output_lab.place(x=500, y=580)
    output_lab.config(text=classes[out[0][0]])

canvas1 = Canvas(height=40,width=949,bg="#FF9900",highlightthickness=0)
canvas1.place(x=0,y=0)

canvas2 = Canvas(height=650,width=15,bg="#FF9900",highlightthickness=0)
canvas2.place(x=0,y=0)

canvas3 = Canvas(height=650,width=20,bg="#FF9900",highlightthickness=0)
canvas3.place(x=250,y=0)

canvas4 = Canvas(height=650,width=15,bg="#FF9900",highlightthickness=0)
canvas4.place(x=935,y=0)

canvas5 = Canvas(height=15,width=949,bg="#FF9900",highlightthickness=0)
canvas5.place(x=0,y=635)

canvas6 = Canvas(height=1,width=200,bg="black",highlightthickness=0)
canvas6.place(x=50,y=55)


#LABELS

my=Label(text="     Cervical Cancer Detection System",font=("Times New Roman",18,"italic"),fg="black",bg="#FF9900")
my.place(x=260,y=3)

my1=Label(text="Menu",font=("Times New Roman",12,"italic"),fg="black",bg="#0099CC")
my1.place(x=22,y=42)

my2=Label(text="Browse-image",font=("Times New Roman",15,"italic"),fg="black",bg="#0099CC")
my2.place(x=330,y=270)

my3=Label(text="Gray-Scale",font=("Times New Roman",15,"italic"),fg="black",bg="#0099CC")
my3.place(x=560,y=270)

my4=Label(text="Histogram Equalization",font=("Times New Roman",15,"italic"),fg="black",bg="#0099CC")
my4.place(x=720,y=270)

my5=Label(text="Equalized Coloured",font=("Times New Roman",15,"italic"),fg="black",bg="#0099CC")
my5.place(x=390,y=525)

my6=Label(text="Cancereous Cell",font=("Times New Roman",15,"italic"),fg="black",bg="#0099CC")
my6.place(x=640,y=525)

#BUTONS

browse_b=Button(text="Browse Input Image",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",
                width=20,font=("Times New Roamn",12,"italic"),fg="black",command=browse)
browse_b.place(x=36,y=80)

browse_b=Button(text="Gray-Scaling",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=gray)
browse_b.place(x=36,y=160)

browse_b=Button(text="Histogram Equalization",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=histogram)
browse_b.place(x=36,y=240)

browse_b=Button(text="Equalized Coloured",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=coloredHist)
browse_b.place(x=36,y=320)

browse_b=Button(text="Detect Cancerous cell",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=EdgeDetect)
browse_b.place(x=36,y=400)

browse_b=Button(text="Output",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=CancerDetect)
browse_b.place(x=36,y=480)

browse_b=Button(text="Exit",bg="#FF9900",highlightthickness=0,activebackground="#48AAAD",width=20,font=("Times New Roamn",12,"italic"),fg="black",command=close)
browse_b.place(x=36,y=560)

#create function that opens file and take path of wholeimage and put in Image.open and place it in canvas
window.mainloop()