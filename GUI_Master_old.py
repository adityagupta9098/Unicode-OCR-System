import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
# import CNNModel 
import sqlite3
import tensorflow as tf
from tensorflow.keras import layers, models
#import tfModel_test as tf_test
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


global fn
import requests
fn=""
##############################################+=============================================================
root = tk.Tk()
# root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("UNICODE OCR SYSTEM")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('a2.jpeg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0) #, relwidth=1, relheight=1)
#
lbl = tk.Label(root, text="UNICODE OCR SYSTEM", font=('times', 35,' bold '), width=40,height=1,bg="Red3",fg="white")
lbl.place(x=0, y=0)


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)

frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="SeaGreen1")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=100)

msg="UNICODE OCR SYSTEM"

def update_label1(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=650)
    
    
    
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
def update_cal(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=350, y=350)
    
    
    
###########################################################################
# def train_model():
 
#     update_label("Model Training Start...............")
    
#     start = time.time()

#     X= CNNModel.main()
#     print(X)
    
#     end = time.time()
        
#     ET="Execution Time: {0:.4} seconds \n".format(end-start)
    
#     msg="Model Training Completed.."+'\n'+ X + '\n'+ ET

#     update_label(msg)

import functools
import operator


def convert_str_to_tuple(tup):
    s = functools.reduce(operator.add, (tup))
    return s

def test_model_proc(fn):
    from keras.models import load_model


    
    
    IMAGE_SIZE = 64
    LEARN_RATE = 1.0e-4
    CH=3
    print(fn)
    if fn != "":
        # Model Architecture and Compilation
       
        model = load_model('OCR_model.h5')
            
        # adam = Adam(lr=LEARN_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        # model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        img = Image.open(fn)
        img = img.resize((IMAGE_SIZE,IMAGE_SIZE))
        img = np.array(img)
        
        
        

        
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        
        img = img.astype('float32')
        img = img / 255.0
        print('img shape:',img)
        prediction = model.predict(img)
        print(np.argmax(prediction))
        yoga=np.argmax(prediction)
        print(yoga)
        
        
                
        # myfile = open("1.txt")
        # txt = myfile.read()
        # print(txt)
        # myfile.close()
                
        
        
        if yoga == 0:
            Cd="Image Not Detected"
            
        elif yoga == 1:
            Cd="Image Detected"
        
            
        A=Cd
        return A
    img = open(fn)    
    myfile = open(r"F:\data\testing_data")
    if img == myfile:
          print(myfile)




# def clear_img():
    
#     img11 = tk.Label(frame_display, background='lightblue4',width=160,height=120)
#     img11.place(x=0, y=0)

def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=40, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=300, y=420)


def test_model():
    global fn
    if fn != "":
    #     update_label("Model Testing Start...............")
    #
    #     start = time.time()
    #
    #     X = test_model_proc(fn)
    #
    #     X1 = "Selected Image is {0}".format(X)
    #
    #     end = time.time()
    #
    #     ET = "Execution Time: {0:.4} seconds \n".format(end - start)
    #
    #     msg = "Image Testing Completed.." + '\n' + X1 + '\n' + ET
    #     fn = ""
    # else:
    #     msg = "Please Select Image For Prediction...."
    #
    # update_label("msg")
    
          predicted_char = predict_character(fn) 
    
          update_label("Predicted Character is :"+predicted_char)  
    
OCR_model = load_model('OCR_model.h5')
def predict_character(image_path):
      img_size = (28, 28)
      img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
      img_array = tf.keras.preprocessing.image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0) / 255.0

      prediction =OCR_model.predict(img_array)
      predicted_class = np.argmax(prediction)
       
      
      img_size = (28, 28)  # Adjust the size based on your dataset
      batch_size = 32
      dataset_path = r"F:\data\training_data"
      # Data preprocessing and augmentation
      train_datagen = ImageDataGenerator(
          rescale=1./255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=False,
          validation_split=0.2
      )

      train_generator = train_datagen.flow_from_directory(
          dataset_path,
          target_size=img_size,
          batch_size=batch_size,
          class_mode='categorical',
          subset='training'
      )

     


        
      # Convert class index to character
      characters = {v: k for k, v in train_generator.class_indices.items()}
      predicted_character = characters[predicted_class]
      return predicted_character
       
    

def openimage():
   
    global fn
    print(fn)
    fileName = askopenfilename(initialdir=r"F:\data\testing_data", title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


#
#        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
#
#        gs = cv2.resize(gs, (x1, y1))
#
#        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root,text='Original',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250)
    #result_label1.place(x=300, y=100)
    img.image = imgtk
    img.place(x=300, y=100)
   # out_label.config(text=imgpath)

def convert_grey():
    global fn 
    print(fn)
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root,text='Gray',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)
    #label_l1 = tk.Label(root, text='Gray' ,compound='bottom', width=4, height=1)
    #label_l1.place(x=690, y=110)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root,text='Binary',font=('times new roman', 20 ,'bold'), image=imgtk,compound='bottom', height=250, width=250)
    img3.image = imgtk
    img3.place(x=880, y=100)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)
    
    
    # def percentage(Width, Depth):
    #   return 100 * float(Width)/float(Depth)

    # print(percentage(5, 7))
    # # If you want to limit the number of decimal to 2, change the number in {:.2f} as you wish;
    # print('{:.2f}'.format(percentage(5, 7)))
    # your_value = 1/3.0
    # print('{:.1%}'.format(your_value)) # Change the "1" to however many decimal places you need
    # # Result:
    # # '33.3%'


#################################################################################################################
def window():

    root.destroy()
    




button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=50)

button2 = tk.Button(frame_alpr, text="Image_preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button2.place(x=10, y=120)

#button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
#button3.place(x=10, y=160)
#
button4 = tk.Button(frame_alpr, text="CNN_Prediction", command=test_model,width=15, height=1,bg="white",fg="black", font=('times', 15, ' bold '))
button4.place(x=10, y=190)
#
#
#

exit = tk.Button(frame_alpr, text="Exit", command=window, width=15, height=1, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=10, y=260)



root.mainloop()