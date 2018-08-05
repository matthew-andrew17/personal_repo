
# coding: utf-8

# In[2]:


import serial
import time
import os
import numpy as np
import subprocess
import multiprocessing


# In[3]:


filename='"C:/Program Files (x86)/digiCamControl/CameraControlCmd.exe" /filename "C:/Users/y7mandre/Pictures/digiCamControl/Session1/test.jpg" /capture'
localfilename="CameraControlCmd.exe /filename test.jpg /capture"
localfilename2="CameraControlRemoteCmd.exe /c capture "


# In[4]:

cd "C:\Program Files (x86)\digiCamControl"

s1 = serial.Serial('COM5',9600)


# In[6]:


def moveto(x,speed=5000):
    command='G1X' + str (x) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    pos_x=x
    print('X axis now at: ' + str(pos_x))

def jog(x,speed=5000):
    global pos_xS
    pos_x=pos_x+x
    command='G1X' + str (pos_x) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    print('X axis now at: ' + str(pos_x))
    
def resetaxes():
    global pos_x
    command='G92 X0 Y0 Z0\n'
    s.write(command.encode())
    pos_x=0
    print('X axis now at: ' + str(pos_x))
    
def trigger(waittime=0.5):

    command="00dw0008HIGH;"
    s1.write(command.encode())
    time.sleep(waittime/2)
    #print("high")
    command="00dw0008 LOW;"
    s1.write(command.encode())
    time.sleep(waittime/2)
    #print("low")
    
def multitrigger(n=100):
    for x in range(n):
        trigger()
        #print(x)

        if np.remainder(x,50)==0:
            s1.reset_input_buffer()
            s1.reset_output_buffer()
            time.sleep(1)



parent_conn, child_conn = multiprocessing.Pipe()
p=multiprocessing.Process(multitrigger(10))
p.start()
p.join() # block until the process is complete - this should be pushed to the end of your script / managed differently to keep it async :)

print ("this will run before finishing") # will tell you when its done.


# In[21]:

