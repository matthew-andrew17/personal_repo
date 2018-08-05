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
from skimage import transform
from PIL import ImageTk, Image
import micro_functions as mf
import tkinter
import tkinter.messagebox
import tkinter.filedialog
import shared_variables as share
import hdf_functions as myh5

C=0
def gui_main():
    global C
    jobs_mf = bg.BackgroundJobManager()
    jobs_mf.new(mf.comlistener)
    mf.cyclecoms()
    mf.startupcommands()
    mf.resetaxes()
    mf.initialize_sharedvariables()
    top = tkinter.Tk() 
	
    top.title('Wolffia')
    
    top.geometry("600x600")
    top.state('zoomed')
    top.update()
    w, h = top.winfo_width(), top.winfo_height()

    def testCallBack():
        text=jog_entry.get()
        tkinter.messagebox.showinfo( "Hello Python", text)
                
    def update():
        x_pos_label_r.config(text=share.x)
		
    def ybuttonp():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.jog_y(text,speed=speedi)

    def ybuttonn():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.jog_y(-text,speed=speedi)

    def xbuttonp():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))		
        mf.jog_x(text,speed=speedi)

    def xbuttonn():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.jog_x(-text,speed=speedi)

    def zbuttonp():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.jog_z(text,speed=speedi)

    def zbuttonn():
        text=jog_entry.get()
        text=np.absolute(float(text))
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.jog_z(-text,speed=speedi)

    def movexbutton():
        text=float(movex_entry.get())
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto_x(text,speed=speedi)
        
    def moveybutton():
        text=float(movey_entry.get())
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto_y(text,speed=speedi)
        
    def movezbutton():
        text=float(movez_entry.get())
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto_z(text,speed=speedi)
        
    def moveallbutton():
        textx=float(movex_entry.get())
        texty=float(movey_entry.get())
        textz=float(movez_entry.get())
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto(textx,texty,textz,speed=speedi)

    def close_coms_button_c():
        top.destroy()
        mf.closecoms()
        gui_starter()

    def cycle_coms_button_c():
        mf.cyclecoms()
        mf.startupcommands()
        mf.resetaxes()

    def resetax_button():
        mf.resetaxes()

    def movehome_button():
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto(0,0,0,speed=speedi)

    def imnav_u():
        C.yview_scroll(-1,"units")
        a=C.yview()
        share.picy=int(share.im_full_bbox[3]*a[0])
        im_list_creator()
        image_creator()
		
    def imnav_d():
        C.yview_scroll(1,"units")
        a=C.yview()
        share.picy=int(share.im_full_bbox[3]*a[0])
        im_list_creator()
        image_creator()
		
    def imnav_l():
        C.xview_scroll(-1,"units")
        a=C.xview()
        share.picx=int(share.im_full_bbox[2]*a[0])
        im_list_creator()
        image_creator()
		
    def imnav_r():
        C.xview_scroll(1,"units")
        a=C.xview()
        share.picx=int(share.im_full_bbox[2]*a[0])
        im_list_creator()
        image_creator()

    def dc_move(event):
        x=C.canvasx(event.x)
        x=(x-share.dispx)/share.im_full_bbox[2]
        C.xview_moveto(x)
        y=C.canvasy(event.y)
        y=(y-share.dispy)/share.im_full_bbox[3]
        C.yview_moveto(y)
        a=C.xview()
        share.picx=int(share.im_full_bbox[2]*a[0])
        a=C.yview()
        share.picy=int(share.im_full_bbox[3]*a[0])
        im_list_creator()
        image_creator()        
		
    def askdirectory():
        share.fnold=share.fn
        share.fn=tkinter.filedialog.askopenfilename()
        if share.fn!=share.fnold:
            myh5.initialize_projectfile(share.fn)
            share.im_positions,share.im_cent_positions=myh5.get_positions()
            share.im_c_list=share.hf.indices
            share.im_current_list=[]
            share.im_current_list_new=[]
            C.delete('all')
            share.imageIDs=[]
            share.imageIDs2=np.zeros(10)
            C.update()

            im_list_creator()
            image_creator()
            directory_label2_r.config(text=share.fn)
            #print('test')

    def saveas_directory():
        share.fnold=share.fn
        share.fn=tkinter.filedialog.asksaveasfilename()
        if share.fn!=share.fnold:
            myh5.initialize_projectfile(share.fn,typer='w')
            share.im_positions,share.im_cent_positions=myh5.get_positions()
            share.im_c_list=share.hf.indices
            share.im_current_list=[]
            share.im_current_list_new=[]
            C.delete('all')
            share.imageIDs=[]
            share.imageIDs2=np.zeros(10)
            C.update()

            im_list_creator()
            image_creator()
            directory_label_r.config(text=share.fn)
			
			
    def scroll_start(event):
        C.scan_mark(event.x, event.y)
    def scroll_move(event):
        C.scan_dragto(event.x, event.y, gain=1)
    def scroll_end(event):
        a=C.xview()
        share.picx=int(share.im_full_bbox[2]*a[0])
        a=C.yview()
        share.picy=int(share.im_full_bbox[3]*a[0])
        C.update()
        im_list_creator()
        image_creator()
        C.update()

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
            share.imgID=C.create_image(share.im_positions[share.im_c_list[x]][0],share.im_positions[share.im_c_list[x]][1],anchor="nw",image=share.imgtk)
            
            share.imageIDs[share.im_c_list[x]]=[share.im_c_list[x],share.imgID,share.imgtk]
			
            share.imageIDs2[share.im_c_list[x]]=share.imgID
			
        if len(share.im_r_list)!=0:
            for x in range(len(share.im_r_list)):
                C.delete(share.imageIDs2[share.im_r_list[x]])
                share.imageIDs[share.im_r_list[x]]=[0,0,0]
        share.im_c_list=[]
        share.im_r_list=[]
        C.update()
			
    def im_list_creator():
        imagenumber=len(list(share.hf['imagedata/'+str(share.zoomlevel)]))
        #print(imagenumber)
        a=C.xview()[0]*share.im_full_bbox[2]+share.dispx
        b=C.yview()[0]*share.im_full_bbox[3]+share.dispy
        share.currentposition=[a,b]
        #print(share.currentposition)
        dist=np.sqrt(np.sum(((share.im_cent_positions-share.currentposition)**2),axis=1))
        argswhere=(np.argwhere(dist<(share.viewportdims[0]*1))).flatten()
        share.im_current_list_new=argswhere
        share.im_c_list=np.setdiff1d(share.im_current_list_new,share.im_current_list)
        #print(share.im_c_list)
        share.im_r_list=np.setdiff1d(share.im_current_list,share.im_current_list_new)
        share.im_current_list=share.im_current_list_new

    def zoomin():
        
        share.im_current_list=[]
        share.im_current_list_new=[]

        if share.zoomlevel>0:
            share.zoomlevel=share.zoomlevel-1
        share.im_positions,share.im_cent_positions=myh5.get_positions()
        a=C.xview()
        b=a[0]*2+share.dispx/share.canvas_bbox[2]
        C.xview_moveto(b)

        a=C.yview()
        b=a[0]*2+share.dispy/share.canvas_bbox[3]
        C.yview_moveto(b)
        zoom_label.config(text=share.zoomlevel)        
        C.delete('all')
        share.imageIDs=[]
        share.imageIDs2=np.zeros(10)
        C.update()
        im_list_creator()
        image_creator()
        C.update()
		
    def zoomout():
        share.im_current_list=[]
        share.im_current_list_new=[]

        if share.zoomlevel<9:
            share.zoomlevel=share.zoomlevel+1
        share.im_positions,share.im_cent_positions=myh5.get_positions()
        a=C.xview()

        b=(a[0]-share.dispx/share.canvas_bbox[2])/2
        C.xview_moveto(b)
        a=C.yview()
        b=(a[0]-share.dispy/share.canvas_bbox[3])/2
        C.yview_moveto(b)
        C.delete('all')
        share.imageIDs=[]
        share.imageIDs2=np.zeros(10)
        zoom_label.config(text=share.zoomlevel)
        C.update()
        im_list_creator()
        image_creator()

    def trigger():
        share.pixelsize=pixel_size_ent.get()
        image,filename=mf.get_newimage()
        myh5.insert_single_image(image,filename)
        share.im_positions,share.im_cent_positions=myh5.get_positions()
        im_list_creator()
        image_creator()
        C.update()

    def mult_trigger():
        number=mult_trigger_ent.get()
        number=np.absolute(int(number))
        mf.multitrigger(n=number)        
	
    def scan_area():
        share.pixelsize=float(pixel_size_ent.get())
        image,filename=mf.get_newimage()
        #print("filename: ", filename)
        filenumber=filename.split('\\')
        filenumber=filenumber[-1]
        filenumber=filenumber.split('.')[0].split('_')[1]
        filenumber=int(filenumber)
        #print("filenumber: ", filenumber)
        share.current_imagename=filename
        del share.todo_dict[filename]
        share.fov[0]=image.shape[1]*share.pixelsize
        share.fov[1]=image.shape[0]*share.pixelsize
        share.scanbound[0]=[define_entry_tl.get(),define_entry_tr.get()]
        share.scanbound[1]=[define_entry_bl.get(),define_entry_br.get()]
        mf.create_scanlist()
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        share.scanjob=jobs_mf.new(mf.scan_area,kw=dict(speedf=speedi,largestind=filenumber))
        #print('test',str(share.scanjob.finished))
        while share.scanjob.finished==False:
            x=0
            y=0
            time.sleep(0.1)
            filenames=list(share.todo_dict.keys())
            #print(filenames)
            for x in range(len(filenames)):
                while y==0:
                    #print(x,filenames[x])
                    try:
                        imagedata=skimage.io.imread(filenames[x])
                        os.remove(filenames[x])
                        y=1
                        #print(y)
                    except:
                        y=0
                        #print(y)
                if correct_images.get()==True:
                    if (share.image_corrector==0).all()!=True:
                        imagedata=imagedata/share.image_corrector
                        imagedata=np.clip((imagedata*200),0,255).astype('uint8')
                myh5.insert_single_image(imagedata,filenames[x])
                y=0
            job_label.config(text=str(share.scanjob.finished))
            im_list_creator()
            image_creator()
            frame1.update()
            
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

            if correct_images.get()==True:
                if (share.image_corrector==0).all()!=True:
                    imagedata=imagedata/share.image_corrector
                    imagedata=np.clip((imagedata*200),0,255).astype('uint8')
				
            myh5.insert_single_image(imagedata,filenames[x])
            im_list_creator()
            image_creator()
        #share.hfjob=jobs_mf.new(mf.cleanup)
		
        speedi=speed_entry.get()
        speedi=np.absolute(float(speedi))
        mf.moveto(0,0,0,speed=speedi)

    def af_g_nomove():
        currentx=share.x
        currenty=share.y
        currentz=share.z
        speedi=speed_entry.get()
        speedi=float(speedi)
        mf.autofocus([currentx,currenty],zstart=currentz-10,zend=currentz+10,number=10,speeda=speedi)
		
    def closehf():
        myh5.closeproj()
	
    def load_imagecorrector():
        filename=tkinter.filedialog.askopenfilename()
        share.image_corrector=skimage.io.imread(filename)
		
    #Declare Variables
	
    correct_images=tkinter.BooleanVar()
		
    frame1=tkinter.Frame(top,relief="raised",borderwidth=3,padx=10)
    frame1.place(x=5,y=5,height=h-10,width=450)
    
    UP = tkinter.Button(frame1, text ="Y+", command = ybuttonp, width='10')
    UP.grid(row=0,column=2,pady=10,sticky='we')

    LEFT = tkinter.Button(frame1, text ="X-", command = xbuttonn)
    LEFT.grid(row=1,column=1,pady=10,sticky='we')

    RIGHT = tkinter.Button(frame1, text ="X+", command = xbuttonp, width='10')
    RIGHT.grid(row=1,column=3,pady=10,sticky='we')
    
    DOWN = tkinter.Button(frame1, text ="Y-", command = ybuttonn)
    DOWN.grid(row=2,column=2,pady=10,sticky='we')
	
    Zp= tkinter.Button(frame1, text ="Z+", command = zbuttonp)
    Zp.grid(row=0,column=0,pady=10,sticky='we')
	
    Zn= tkinter.Button(frame1, text ="Z-", command = zbuttonn)
    Zn.grid(row=2,column=0,pady=10,sticky='we')
    
    jog_entry_label=tkinter.Label(frame1,text='Jog Amount:')
    jog_entry_label.grid(row=5,column=0)
    
    jog_entry=tkinter.Entry(frame1,width='10')
    jog_entry.insert(0,'10')
    jog_entry.grid(row=5, column=1)

    speed_entry_label=tkinter.Label(frame1,text='Speed:')
    speed_entry_label.grid(row=5,column=2)
    
    speed_entry=tkinter.Entry(frame1,width='10')
    speed_entry.insert(0,'500')
    speed_entry.grid(row=5, column=3)

    frame1.grid_rowconfigure(6, minsize=15)
    
    movex_entry_label=tkinter.Label(frame1,text='Move X to:')
    movex_entry_label.grid(row=7,column=0)
    
    movex_entry=tkinter.Entry(frame1,width='10')
    movex_entry.insert(0,'0')
    movex_entry.grid(row=7, column=1)
    
    movex_button = tkinter.Button(frame1, text ="X", command = movexbutton)
    movex_button.grid(row=7,column=2,sticky='we')
    
    movey_entry_label=tkinter.Label(frame1,text='Move Y to:')
    movey_entry_label.grid(row=8,column=0)
    
    movey_entry=tkinter.Entry(frame1,width='10')
    movey_entry.insert(0,'0')
    movey_entry.grid(row=8, column=1)
    
    movey_button = tkinter.Button(frame1, text ="Y", command = moveybutton)
    movey_button.grid(row=8,column=2,sticky='we')
    
    movez_entry_label=tkinter.Label(frame1,text='Move Z to:')
    movez_entry_label.grid(row=9,column=0)
    
    movez_entry=tkinter.Entry(frame1,width='10')
    movez_entry.insert(0,'0')
    movez_entry.grid(row=9, column=1)
    
    movez_button = tkinter.Button(frame1, text ="Z", command = movezbutton)
    movez_button.grid(row=9,column=2,sticky='we')
    
    moveall_button=tkinter.Button(frame1, text ="Move\nAll", command = moveallbutton, width=10)
    moveall_button.grid(row=7,column=3,sticky='nswe',rowspan='3')
    
    frame1.grid_rowconfigure(10, minsize=15)

	
    x_pos_label=tkinter.Label(frame1,text='X Position:')
    x_pos_label.grid(row=11,column=0)

    x_pos_label_r=tkinter.Label(frame1,text=share.x)
    x_pos_label_r.grid(row=11,column=1)
	
    y_pos_label=tkinter.Label(frame1,text='Y Position:')
    y_pos_label.grid(row=12,column=0)

    y_pos_label_r=tkinter.Label(frame1,text=share.y)
    y_pos_label_r.grid(row=12,column=1)
	
    z_pos_label=tkinter.Label(frame1,text='Z Position:')
    z_pos_label.grid(row=13,column=0)

    z_pos_label_r=tkinter.Label(frame1,text=share.z)
    z_pos_label_r.grid(row=13,column=1) 
	
    frame1.grid_rowconfigure(14, minsize=15)

    coms_label=tkinter.Label(frame1,text='Com Interface:')
    coms_label.grid(row=15,column=0)

    close_coms_button=tkinter.Button(frame1, text ="Close Coms", command = close_coms_button_c)
    close_coms_button.grid(row=16,column=0,padx=10,sticky='we')

    cycle_coms_button=tkinter.Button(frame1, text ="Cycle Coms", command = cycle_coms_button_c)
    cycle_coms_button.grid(row=16,column=1,padx=10,sticky='we')

    frame1.grid_rowconfigure(17, minsize=15)

    axes_control_label=tkinter.Label(frame1,text='Axes Control:')
    axes_control_label.grid(row=15,column=2)

    axes_control_button=tkinter.Button(frame1, text ="Move Home", command = movehome_button)
    axes_control_button.grid(row=16,column=2,padx=2,sticky='we')

    axes_control_button2=tkinter.Button(frame1, text ="Define Home", command = resetax_button)
    axes_control_button2.grid(row=16,column=3,padx=2,sticky='we')
	
#    ut_port=tkinter.Label(frame1,text=str(share.picx),width='10',anchor='w')
#    ut_port.grid(row=17,column=0)
#    ut_port2=tkinter.Label(frame1,text=str(share.picy),width='10',anchor='w')
#    ut_port2.grid(row=17,column=1)

		
    frame2=tkinter.Frame(top,relief="raised",borderwidth=3)
    frame2.place(x=460,y=5,height=(h-10),width=(w-465))
	
    directory_label=tkinter.Label(frame1,text='Select Project File:',anchor='w')
    directory_label.grid(row=17,column=0,columnspan='2')	
	
    directory_button=tkinter.Button(frame1,text='Open',command=askdirectory,width='10')
    directory_button.grid(row=18,column=0)
	
    directory_label_r=tkinter.Label(frame1,text=share.fn,anchor='w',width=30)
    directory_label_r.grid(row=18,column=1,columnspan='3')	
	
    directory_button2=tkinter.Button(frame1,text='Save As',command=saveas_directory,width='10')
    directory_button2.grid(row=19,column=0)
	
    directory_label2_r=tkinter.Label(frame1,text=share.fn,anchor='w',width=30)
    directory_label2_r.grid(row=19,column=1,columnspan='3')	

    trigger_b=tkinter.Button(frame1,text='Trigger',command=trigger,width='10')
    trigger_b.grid(row=20,column=0)
	
    mult_trigger_b=tkinter.Button(frame1,text='Multi Trig',command=mult_trigger,width='10')
    mult_trigger_b.grid(row=21,column=0)
	
    mult_trigger_ent=tkinter.Entry(frame1,width='10')
    mult_trigger_ent.insert(0,'10')
    mult_trigger_ent.grid(row=21, column=1)
	
    pixel_size_label=tkinter.Label(frame1,text='Pix Size/um',anchor='w',width=10)
    pixel_size_label.grid(row=21,column=2)	
	
    pixel_size_ent=tkinter.Entry(frame1,width='10')
    pixel_size_ent.insert(0,'0.005')
    pixel_size_ent.grid(row=21, column=3)
	
    define_label=tkinter.Label(frame1,text='Def Bounds',anchor='w',width=10)
    define_label.grid(row=22,column=0)	

    define_entry_tl=tkinter.Entry(frame1,width='10')
    define_entry_tl.insert(0,'0')
    define_entry_tl.grid(row=23, column=0)

    define_entry_tr=tkinter.Entry(frame1,width='10')
    define_entry_tr.insert(0,'0')
    define_entry_tr.grid(row=23, column=1)

    define_entry_bl=tkinter.Entry(frame1,width='10')
    define_entry_bl.insert(0,'0')
    define_entry_bl.grid(row=24, column=0)

    define_entry_br=tkinter.Entry(frame1,width='10')
    define_entry_br.insert(0,'0')
    define_entry_br.grid(row=24, column=1)
	
    scan_area_b=tkinter.Button(frame1,text='Scan Area',command=scan_area,width='10')
    scan_area_b.grid(row=23,column=2)
	
    close_hfile=tkinter.Button(frame1, text ="Close P File", command = closehf)
    close_hfile.grid(row=24,column=2,padx=2,sticky='we')

    job_label=tkinter.Label(frame1,text=str(share.scanjob),anchor='w',width=10)
    job_label.grid(row=25,column=0)	
	
    af_button=tkinter.Button(frame1, text ="Autofocus", command = af_g_nomove)
    af_button.grid(row=26,column=0,padx=2,sticky='we')
	
    corrector_load=tkinter.Button(frame1, text ="Im Corrector", command = load_imagecorrector)
    corrector_load.grid(row=26,column=1,padx=2,sticky='we')
	
    #corrector_label=tkinter.Label(frame1,text='Use Corrector:')
    #corrector_label.grid(row=26,column=2)
	
    corrector_checkbox=tkinter.Checkbutton(frame1,variable=correct_images,text='Use Correct?',onvalue=True,offvalue=False)
    corrector_checkbox.grid(row=26,column=2)	
	
    frame1.columnconfigure(1,weight=0)
    
    frame1.update()
    frame2.update()
    
    f2w, f2h = frame2.winfo_width(), frame2.winfo_height()
    
    C = tkinter.Canvas(frame2, bg="blue", height=(f2h-50), width=(f2w-20),relief="raised",borderwidth=1, scrollregion=(0,0,share.im_full_bbox[2],share.im_full_bbox[3]))
    C.grid(row=0,columnspan=10,padx=5,pady=5)
    C.update()
    

    Cw, Ch = C.winfo_width(), C.winfo_height()
    share.viewportdims=[Cw,Ch]
    share.dispx=int(Cw/2)
    share.dispy=int(Ch/2)
	
    C.bind("<ButtonPress-1>", scroll_start)
    C.bind("<B1-Motion>", scroll_move)
    C.bind("<ButtonRelease-1>", scroll_end)
    
    c_px,c_py=C.winfo_pointerxy()
    tlx=C.winfo_rootx()
    tly=C.winfo_rooty()
    transx=tlx+share.dispx
    transy=tly+share.dispy
		
    C.bind('<Double-1>', dc_move)
    
    mousex_label=tkinter.Label(frame2,text='X:')
    mousex_label.grid(row=1,column=0)
    
    mousex_label_val=tkinter.Label(frame2,text=c_px-transx+share.picx)
    mousex_label_val.grid(row=1,column=1)    
    
    mousey_label=tkinter.Label(frame2,text='Y:')
    mousey_label.grid(row=1,column=2)
    
    mousey_label_val=tkinter.Label(frame2,text=c_py-transy+share.picy)
    mousey_label_val.grid(row=1,column=3) 
    
    mousez_label=tkinter.Label(frame2,text='Z:')
    mousez_label.grid(row=1,column=4)
    
    mousez_label_val=tkinter.Label(frame2,text=share.z)
    mousez_label_val.grid(row=1,column=5) 
	
    moveim_b_u=tkinter.Button(C,text="up",command=imnav_u)
    moveim_b_u.place(x=Cw-95,y=Ch-70,anchor="sw",width=40)
	
    moveim_b_d=tkinter.Button(C,text="down",command=imnav_d)
    moveim_b_d.place(x=Cw-95,y=Ch-20,anchor="sw",width=40)
	
    moveim_b_l=tkinter.Button(C,text="left",command=imnav_l)
    moveim_b_l.place(x=Cw-140,y=Ch-45,anchor="sw",width=40)
	
    moveim_b_r=tkinter.Button(C,text="right",command=imnav_r)
    moveim_b_r.place(x=Cw-50,y=Ch-45,anchor="sw",width=40)
	
    zoomin_b=tkinter.Button(C,text="in",command=zoomin)
    zoomin_b.place(x=40,y=Ch-120,anchor='center',width=40)
	
    zoomout_b=tkinter.Button(C,text="out",command=zoomout)
    zoomout_b.place(x=40,y=Ch-40,anchor='center',width=40)
	
    zoom_label=tkinter.Label(C,text=share.zoomlevel,bg='white')
    zoom_label.place(x=40,y=Ch-80,anchor='center',width=40)
    
    a=C.xview()
        
    def clock():
        share.correct_images=correct_images.get()
        x_pos_label_r.config(text=share.x)
        y_pos_label_r.config(text=share.y)
        z_pos_label_r.config(text=share.z)
        
        c_px,c_py=C.winfo_pointerxy()			
#        mousex_label_val.config(text=c_px-transx+share.picx)
 #       mousey_label_val.config(text=c_py-transy+share.picy)
        mousex_label_val.config(text=c_px-tlx+share.picx)
        mousey_label_val.config(text=c_py-tly+share.picy)


        moveim_b_l.config(command=imnav_l)
        moveim_b_r.config(command=imnav_r)
        moveim_b_u.config(command=imnav_u)
        moveim_b_d.config(command=imnav_d)
        share.pixelsize=float(pixel_size_ent.get())
		
#        ut_port.config(text=('X: '+ str(share.picx)))
#        ut_port2.config(text=('Y: '+ str(share.picx)))
		
        frame1.after(100, clock)
        
    clock()

    top.mainloop()
	
