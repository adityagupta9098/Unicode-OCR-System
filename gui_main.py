
import tkinter as tk
from PIL import Image, ImageTk


##############################################+=============================================================

#####For background Image
class RegistrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Registration Form")

        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        image2 = Image.open('b1.jpg')
        image2 = image2.resize((w, h), Image.ANTIALIAS)
        self.background_image = ImageTk.PhotoImage(image2)

        self.background_label = tk.Label(root, image=self.background_image)
        self.background_label.place(x=0, y=0)


#################################################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

root = tk.Tk()
app = RegistrationApp(root)

root.attributes('-fullscreen', True)

root.title("UNICODE OCR SYSTEM")
################################$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def reg():
    from subprocess import call
    call(["python","registration.py"])

def log():
    from subprocess import call
    call(["python","login.py"])
    
def window():
  root.destroy()


    
button1 = tk.Button(root, text="Login", command=log, width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button1.place(x=100, y=160)

button2 = tk.Button(root, text="Register",command=reg,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button2.place(x=100, y=240)

button3 = tk.Button(root, text="Exit",command=window,width=14, height=1,font=('times', 20, ' bold '), bg="#152238", fg="white")
button3.place(x=100, y=330)

root.mainloop()


