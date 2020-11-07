# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 08:51:46 2020

@author: ofekf
"""

import numpy as np
from tkinter import *
from PIL import Image, ImageDraw
import network, mnist_loader

canvas_width = 200
canvas_height = 200

x1,y1=None, None

img= Image.new("1",(200,200), color=255)
draw=ImageDraw.Draw(img)

def paint(event):
    global x1
    global y1
    if x1==None:
        python_green = "#476042"
        x1, y1 = ( event.x - 1 ), ( event.y - 1 )
        x2, y2 = ( event.x + 1 ), ( event.y + 1 )
        w.create_oval( x1, y1, x2, y2, fill = python_green )
        draw.point([x1, y1, x2, y2], fill='black')
    else:
        w.create_line(x1,y1,event.x,event.y ,width="7")
        draw.line([x1,y1,event.x,event.y],width=10)
        x1,y1=event.x,event.y

def push(event):
    global x1
    global y1
    x1=None
    y1=None

def delete(event):
    w.delete('all')
    draw.rectangle([0,0,400,400], fill='white')
    
def ent(event):
    w.delete('all')
    tmp=img.resize((28,28))
    tmp.save('file.jpg')
    #draw.rectangle([0,0,400,400], fill='white')
    #check=1-(np.asarray(tmp)/255)
    #check=check.reshape((784,1)) 
    #print(np.argmax(net.feedforward(check)))
    
    
#train,val,dump=mnist_loader.load_data_wrapper()
#net=network.Network([784,100,10])
#net.SGD(train,30,10,0.5,lmbda=5,evaluation_data=val)
    
    
master = Tk()
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)

w.bind("<Button-1>", push)
w.bind( "<B1-Motion>", paint )
w.bind("<Button-3>",delete)
w.bind("<Tab>", ent)


message = Label( master, text = "Press and Drag the mouse to draw \n right click to reset, Tab to detect" )
message.pack( side = BOTTOM )
    
mainloop()