import serial
import random
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
from skimage import transform
from PIL import ImageTk, Image
import micro_functions as mf
import tkinter
import tkinter.messagebox
import tkinter.filedialog
import shared_variables as share
import gui_functions as gu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def initialize_projectfile(h5_file,typer='a'):
    share.hf = h5py.File(h5_file,typer)
    for x in range(10):
        share.hf.datagroup=share.hf.require_group('imagedata/'+str(x))
        #print(x)
    share.hf.datagroup=share.hf.require_group('imagedata/'+str(share.zoomlevel))
    share.hf.datalist=list(share.hf.datagroup)
    share.hf.indices=list(range(len(share.hf.datagroup)))
	
def closeproj():
    share.hf.close()
    share.hf = 0

def get_positions():
    positions=np.zeros((len(share.hf.datagroup),2))
    center_positions=np.zeros((len(share.hf.datagroup),2))    
    for x in range(len(list(share.hf['imagedata/'+str(share.zoomlevel)]))):
        positions[x]=np.array(share.hf['imagedata/'+str(share.zoomlevel)+'/image_'+str(x)].attrs['position'])/float(share.hf['imagedata/'+str(share.zoomlevel)+'/image_'+str(x)].attrs['pixelsize'])
        shape=share.hf['imagedata/'+str(share.zoomlevel)+'/image_'+str(x)].shape
        center_positions[x,0]=positions[x,0]+shape[1]/2
        center_positions[x,1]=positions[x,1]+shape[0]/2
    return positions, center_positions
	
def create_pyramid():
    for x in range(1,10):
        share.hf.datafrom=share.hf.get('imagedata/'+str(x-1))
        #print(list(share.hf.datafrom),'imagedata/'+str(x-1))
        share.hf.imagedata=share.hf.get('imagedata')
        share.hf['imagedata'].require_group(str(x))
        del share.hf['imagedata'][str(x)]
        share.hf.datato=share.hf.require_group('imagedata/' + str(x))
        share.hf.datalist=list(share.hf.datafrom)
        for y in range(len(share.hf.datalist)):
            z=np.array(share.hf['imagedata/'+str(x-1)+'/image_'+str(y)])
            imgtk=mf.np2tk_resize(z,2)
            string='image_'+str(y)
            share.hf.datato.create_dataset(string,data=imgtk)
            share.hf['imagedata/'+str(x)+'/image_'+str(y)].attrs['position']=share.hf['imagedata/'+str(x-1)+'/image_'+str(y)].attrs['position']/2
			
def get_largest_hd_image_ind():
    imagelist=list(share.hf['imagedata/0'].keys())
    if len(imagelist)!=0:
        imagelist_ind=np.zeros(len(imagelist))
        for x in range(len(imagelist)):
            imagelist_ind[x]=int(float(imagelist[x].split('_')[1]))
        max_ind=int(np.max(imagelist_ind))
    else:
        max_ind=-1
    return max_ind
	
def insert_single_image(image,filename):
    current_ind=get_largest_hd_image_ind()
    current_ind=current_ind+1
    share.hf.create_dataset('imagedata/0/image_'+str(current_ind),data=image)
    share.hf['imagedata/0/image_'+str(current_ind)].attrs['position']=share.todo_dict.pop(filename)
    share.hf['imagedata/0/image_'+str(current_ind)].attrs['pixelsize']=share.pixelsize
    for x in range(1,10):
        share.hf['imagedata'].require_group(str(x))
        z=np.array(share.hf['imagedata/'+str(x-1)+'/image_'+str(current_ind)])
        imgtk=mf.np2tk_resize(z,2)
        string='image_'+str(current_ind)
        share.hf.create_dataset('imagedata/'+str(x)+'/'+string,data=imgtk)
        share.hf['imagedata/'+str(x)+'/'+string].attrs['position']=share.hf['imagedata/'+str(x-1)+'/'+string].attrs['position']/2
        share.hf['imagedata/'+str(x)+'/'+string].attrs['pixelsize']=share.hf['imagedata/'+str(x-1)+'/'+string].attrs['pixelsize']

		
		
		
		
