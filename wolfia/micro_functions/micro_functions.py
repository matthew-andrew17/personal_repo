import shared_variables as share
import tkinter
import tkinter.messagebox
import serial
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import subprocess
import multiprocessing
from IPython.lib import backgroundjobs as bg
import h5py
import skimage
import sys
from skimage import io
from PIL import ImageTk, Image
import hdf_functions as myh5
import gui_functions as gu
import cv2
from cv2 import MOTION_EUCLIDEAN

def opencoms(port1='COM5',baud1=115200,port2='COM7',baud2='9600'):
    global s
    global s1
    s = serial.Serial(port1,baud1)
    s1 = serial.Serial(port2,baud2)
    time.sleep(1)
    print("Coms Open")
	
def closecoms(port1='COM5',baud1=115200,port2='COM7',baud2='9600'):
    global s
    global s1
    share.list_kill=True
    s.close()
    s1.close()
    time.sleep(1)
    print("Coms Closed")

def cyclecoms():
    global s
    global s1
    share.list_kill=True
    s.close()
    s1.close()
    s = serial.Serial('COM5',115200)
    s1 = serial.Serial('COM7',9600)
    time.sleep(2)
    print("Com Cycle Complete")

def comlistener():
    share.list_kill=False
    while share.list_kill!=True:
        command='?\n'
        s.write(command.encode())
        time.sleep(0.05)
        a=str(mf.s.read_all())
        a=a.split(",")
        status=a[0][3:]
        share.x=a[4][5:]
        share.y=a[5]
        share.z=(a[6].split(">"))[0]
   
def moveto(x,y,z,speed=5000):
    command='G1X' + str (x) + ' Y' + str(y)+ 'Z' + str(z) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    distancex=np.absolute(x-share.x)
    distancey=np.absolute(y-share.y)
    distancez=np.absolute(z-share.z)
    distance=np.sqrt(distancex**2+distancey**2+distancez**2)
    share.x=x
    share.y=y
    share.z=z
    timetaken=60*distance/speed
    time.sleep(timetaken)

def register_images_ECC(image1,image2,warp_matrix=np.eye(2,3,dtype=np.float32),number_of_iterations=500,termination_eps=8e-10,me=cv2.MOTION_AFFINE):
    #warp_matrix = np.eye(2, 3, dtype=np.float32)
    #me=cv2.MOTION_AFFINE
    im1=(skimage.color.rgb2grey(image1)*256).astype('uint8')
    im2=(skimage.color.rgb2grey(image2)*256).astype('uint8')

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    (cc,warpmatrix)=cv2.findTransformECC(im1,im2,warp_matrix,me,criteria)
			
    im2warped=cv2.warpAffine(image2, (warpmatrix), (im2.shape[1],im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im2warped,warpmatrix

def register_images_ORB(im1, im2,MAX_FEATURES=500,GOOD_MATCH_PERCENT=0.15):
 
    #MAX_FEATURES = 500
    #GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
   
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
 
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
 
    # Draw top matches
    #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
   
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
   
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
    return im1Reg, h

'''

TESTING DIRECT SSD REGISTRATION - VERY SLOW AND DIDNT WORK

image1=imlist[1]
image2=imlist[10]

image1_g=cv2.cvtColor(image1,cv2.COLOR_RGB2GRAY)
image2_g=cv2.cvtColor(image2,cv2.COLOR_RGB2GRAY)

def register_images_SSD(x,im1,im2):
    x=x.reshape(2,3)
    im2warped=cv2.warpAffine(im2, (x), (im1.shape[1],im1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    dif=(im1-im2warped)**2
    ssd=np.sum(dif[np.where(im2warped>0)])
    nnz=np.sum(im2warped>0)
    ssd=ssd/nnz
    return ssd

t1=time.time()
difference2=register_images_SSD(warp_matrix0,image1_g,image2_g)
t2=time.time()

jac=np.array(([1,1,10],[1,1,10]), dtype=np.float32)
jac=jac.reshape(-1)

warp_matrix0 = np.array(([1,0,0],[0,1,0]), dtype=np.float32)#np.eye(2, 3, dtype=np.float32)
wp0=np.reshape(warp_matrix0,-1)

wp_opt=scipy.optimize.minimize(register_images_SSD,wp0,args=(image1_g,image2_g),options={'gtol': 1e-5, 'disp': True,'eps':0.1})

t3=time.time()
print(t2-t1,t3-t2)
'''
	
	
def moveto_x(x,speed=5000):
    command='G1X' + str (x) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    distancex=np.absolute(x-share.x)
    share.x=x
    timetaken=60*distancex/speed
    time.sleep(timetaken)

def moveto_y(y,speed=5000):
    command='G1Y' + str (y) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    distancey=np.absolute(y-share.y)
    share.y=y
    timetaken=60*distancey/speed
    time.sleep(timetaken)
	
def moveto_z(z,speed=5000):
    command='G1Z' + str (z) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    distancez=np.absolute(z-share.z)
    share.z=z
    timetaken=60*distancez/speed
    time.sleep(timetaken)
	
def jog(x,y,z,speed=5000):
    distancex=np.absolute(x)
    distancey=np.absolute(y)
    distancez=np.absolute(z)
    distance=np.sqrt(distancex**2+distancey**2+distancez**2)
    timetaken=60*distance/speed
    share.x=share.x+x
    share.y=share.y+y
    share.z=share.z+z
    command='G1X' + str (share.x) + ' Y' + str(share.y)+ 'Z' + str(share.z) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    time.sleep(timetaken)

def jog_x(x,speed=5000):
    distancex=np.absolute(x)
    distance=distancex
    timetaken=60*distance/speed
    share.x=share.x+x
    command='G1X' + str (share.x) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    time.sleep(timetaken)

def jog_y(x,speed=5000):
    distancex=np.absolute(x)
    distance=distancex
    timetaken=60*distance/speed
    share.y=share.y+x
    command='G1Y' + str (share.y) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    time.sleep(timetaken)
	
def jog_z(x,speed=5000):
    distancex=np.absolute(x)
    distance=distancex
    timetaken=60*distance/speed
    share.z=share.z+x
    command='G1Z' + str (share.z) + 'F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    time.sleep(timetaken)
    
def resetaxes():
    command='G92 X0 Y0 Z0\n'
    s.write(command.encode())
    s.read_all()
    share.x=0
    share.y=0
    share.z=0
    
def trigger(waittime=0.5):
    command="00dw0008HIGH;"
    s1.write(command.encode())
    time.sleep(waittime/2)
    #print("high")
    command="00dw0008 LOW;"
    s1.write(command.encode())
    s.read_all()
    time.sleep(waittime/2)

def home_negx(speed=5000):
    command='G1X-1000F' + str(speed) + '\n'
    s.write(command.encode())
    s.read_all()
    time.sleep(10)
    cyclecoms()
    startupcommands()
    resetaxes()
    print('before')

    jog(30)
    print('after')
    cyclecoms()
    startupcommands()
    jog(10)

    resetaxes()
   
def multitrigger(n=100):

    for x in range(n):
        trigger()
#        print(testvar)

        if np.remainder(x,50)==0:
            s1.reset_input_buffer()
            s1.reset_output_buffer()
            time.sleep(1)
            
def movetriggerx(n=10,delx=1,sp=5000):

    for x in range(n):
        trigger()
        jog(delx,speed=sp)

        if np.remainder(x,50)==0:
            s1.reset_input_buffer()
            s1.reset_output_buffer()
            time.sleep(1)
    trigger()
    moveto(0)
    
def startupcommands():
    s.write("\r\n\r\n".encode())
    time.sleep(2)
    s.flushInput()
    s.reset_input_buffer()
    s.reset_output_buffer()
    s.write("$RST=*\n".encode())
    time.sleep(0.1)
    command='$21=1\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$22=1\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$120=100000\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$121=100000\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$122=100000\n'
    s.write(command.encode())
    time.sleep(0.1)

    command='$110=15000\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$100=1300.793462\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$101=1609.056092\n'
    s.write(command.encode())
    time.sleep(0.1)	
    command='$111=15000\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='$112=15000\n'
    s.write(command.encode())
    time.sleep(0.1)
    command='G92 X0 Y0 Z0\n'
    s.write(command.encode())
    command='$X\n'
    s.write(command.encode())
    s.read_all()
    share.x=0
    share.y=0
    share.z=0
    testvar=10
    print("startup commands complete")
    	
def extract_image():
    share.image=share.palette[share.picy:share.picy+share.viewportdims[1],share.picx:share.picx+share.viewportdims[0]]
	
def extract_line():
    share.imageo=list(np.ndarray.flatten(share.image[share.dispy]))
    share.imagen=list(np.ndarray.flatten(share.palette[share.picy+share.dispy,share.picx:share.picx+share.viewportdims[0]]))

def np2tk(image):
    img=ImageTk.PhotoImage(image=Image.fromarray(image))
    return img
	
def np2tk_resize(image,factor):
    #print(image.shape)
    img=Image.fromarray(image).resize((int(image.shape[1]/factor),int(image.shape[0]/factor)),Image.NEAREST)
    return img

def change_digicam_folder(folder='C:\\Users\\y7mandre\\Pictures\\scanner_imagedump'):
    share.temp_folder=folder
    os.chdir('C:\\Program Files (x86)\\digiCamControl\\')
    os.system(r'CameraControlRemoteCmd.exe /c set session.folder '+ folder)
    os.chdir(share.temp_folder)	
	
def get_newimage():
    trigger()
    newimage=[]
    while len(newimage)==0:
        current_filelisto=os.listdir(share.temp_folder)
        time.sleep(0.01)
        current_filelistn=os.listdir(share.temp_folder)
        newimage=np.setdiff1d(current_filelistn,current_filelisto)
    filename=share.temp_folder+'\\'+str(newimage[0])
    x=0
    while x==0:
        try:
            imagedata=skimage.io.imread(filename)
            os.remove(filename)
 
            x=1
        except:
            x=0
    
    share.todo_dict[filename]=[share.x,share.y]
    return imagedata,filename

def create_scanlist():
    topleft=share.scanbound[0]
    bottomright=share.scanbound[1]
    xjog=share.fov[0]*0.8
    xnumber=int(np.ceil((bottomright[0]-topleft[0])/xjog))
    yjog=share.fov[1]*0.8
    ynumber=int(np.ceil((bottomright[1]-topleft[1])/yjog))
    share.scanpositions=np.zeros((xnumber*ynumber,2))
    x=0
    y=0
    n=0
    xcount=0
    modul=1
    while y<ynumber:
        while xcount<xnumber:
            share.scanpositions[n]=[topleft[0]+x*xjog,topleft[1]+y*yjog]
            x=x+modul
            xcount=xcount+1
            n=n+1
        y=y+1
        x=x-modul
        xcount=0
        modul=modul*-1
        
def scan_area(speedf=5000,largestind=0):
    n=share.scanpositions.shape[0]
    #largestind=get_largest_folder_image_ind()
    largestind=largestind+1
    #print(n)
    for x in range(n):
        xpos=share.scanpositions[x,0]
        ypos=share.scanpositions[x,1]
        moveto(xpos,ypos,share.z,speed=speedf)
        trigger()
        fn_string=share.temp_folder + '\\x_'+str(largestind).zfill(4)+'.jpg'
        largestind=largestind+1
        share.todo_dict[fn_string]=[xpos,ypos]

def cleanup():
    while share.scanjob.finished==False:
        filenames=list(share.todo_dict.keys())
        for x in range(len(filenames)):
            while y==0:
                try:
                    imagedata=skimage.io.imread(filenames[x])
                    os.remove(filenames[x])
                    y=1
                except:
                    y=0
            myh5.insert_single_image(imagedata,filenames[x])
    filenames=list(share.todo_dict.keys())
    x=0
    for x in range(len(filenames)):
        while y==0:
            try:
                imagedata=skimage.io.imread(filenames[x])
                os.remove(filenames[x])
                y=1
            except:
                y=0
        myh5.insert_single_image(imagedata,filenames[x])

def image_creator():
    max_im_c=0
    if len(share.im_c_list)!=0:
        max_im_c=max(share.im_c_list)

    for x in range(1+max_im_c-len(share.imageIDs)):
        share.imageIDs.append([0,0,0])

    share.imageIDs2=np.resize(share.imageIDs2,len(list(share.hf['imagedata/'+str(share.zoomlevel)]))).astype('uint64')
		
    for x in range(len(share.im_c_list)):

        share.image=np.array(share.hf['imagedata/'+str(share.zoomlevel)+'/image_'+str(share.im_c_list[x])])
        share.imgtk=mf.np2tk(share.image)
        share.imgID=gu.C.create_image(share.im_positions[share.im_c_list[x]][0],share.im_positions[share.im_c_list[x]][1],anchor="nw",image=share.imgtk)
            
        share.imageIDs[share.im_c_list[x]]=[share.im_c_list[x],share.imgID,share.imgtk]
			
        share.imageIDs2[share.im_c_list[x]]=share.imgID
			
    if len(share.im_r_list)!=0:
        for x in range(len(share.im_r_list)):
            gu.C.delete(share.imageIDs2[share.im_r_list[x]])
            share.imageIDs[share.im_r_list[x]]=[0,0,0]
    share.im_c_list=[]
    share.im_r_list=[]
    gu.C.update()
			
def im_list_creator():
    imagenumber=len(list(share.hf['imagedata/'+str(share.zoomlevel)]))
    #print(imagenumber)
    a=gu.C.xview()[0]*share.im_full_bbox[2]+share.dispx
    b=gu.C.yview()[0]*share.im_full_bbox[3]+share.dispy
    share.currentposition=[a,b]
    #print(share.currentposition)
    dist=np.sqrt(np.sum(((share.im_cent_positions-share.currentposition)**2),axis=1))
    argswhere=(np.argwhere(dist<(share.viewportdims[0]*1))).flatten()
    share.im_current_list_new=argswhere
    share.im_c_list=np.setdiff1d(share.im_current_list_new,share.im_current_list)
    #print(share.im_c_list)
    share.im_r_list=np.setdiff1d(share.im_current_list,share.im_current_list_new)
    share.im_current_list=share.im_current_list_new

		
def get_largest_folder_image_ind():
    imagelist=os.listdir(share.temp_folder)
    if len(imagelist)!=0:
        imagelist_inds=np.zeros(len(imagelist))
        for x in range(len(imagelist)):
            imagelist_inds[x]=int((imagelist[x].split('_')[1]).split('.')[0])
        maxind=(int(np.max(imagelist_inds)))
    else:
        maxind=0
    return maxind
        		
def autofocus(position,zstart=-1,zend=1,number=10,speeda=500):
    zav=(zstart+zend)/2
    zrange=abs(zend-zstart)
    print(position[0],position[1],zav,speeda)
    moveto(position[0],position[1],zav,speed=speeda)
    zincr=zrange/number
    for x in range(number):
        zpos=zstart+x*zincr
        moveto(position[0],position[1],zpos,speed=speeda)
        trigger()
    moveto(position[0],position[1],zav,speed=speeda)
	
def initialize_sharedvariables():
    share.x=0
    share.y=0
    share.z=0
    share.picx=0
    share.picy=0
    share.picz=0
    share.list_kill=True
    share.image=0
    share.pallete=0
    share.viewportdims=[0,0]
    share.fn=0
    share.im_full_bbox=[0,0,25000,25000]
    share.canvas_bbox=[0,0,25000,25000]
    share.palette_shape=[0,0]
    share.hf=0 #this is the project file
    share.im_positions=0
    share.im_cent_positions=0
    share.c_im_xind=0
    share.c_im_yind=0
    share.imagelist=[]
    share.imageIDs=[]
    share.imageIDs2=np.zeros(100)
    #share.zoomlevel=2
    share.im_c_list=[]
    share.im_current_list=[]
    share.im_current_list_new=[]
    share.currentposition=[0,0]