################################### Version on the GitHub collaborated with Jeff ##################################
###################################################################################################################


###################################################################################################################
####################################### Last modified on Apr 20 14:43:15 ##########################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

import numpy as np
import h5py
import os
import copy
import cv2
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
# from skimage.io import imread, imshow
import pyclesperanto_prototype as cle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from shapely.geometry import Polygon, mapping, MultiPoint
from NeuronDiagnosis import *
################################### To show the poly on the subplot ##################################


def coords_in_poly(vertices):
    # vertices = [[0, 0], [4, 0], [4, 4], [0, 4]]
    ### get all the pixel in side the polygon
    if len(vertices)>3:
        p = Polygon(vertices)
        if len(p.bounds)==4:
            xmin, ymin, xmax, ymax = p.bounds
            ## the following number 1 can determine the number of grid 
            x = np.arange(np.floor(xmin), np.ceil(xmax) + 1)  # array([0., 1., 2.])
            y = np.arange(np.floor(ymin), np.ceil(ymax) + 1)  # array([0., 1., 2.])
            points = MultiPoint(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]))
            a = points.intersection(p)
            coords = np.array([(p.y, p.x) for p in a.geoms])
            # coords = np.array([(p.y, p.x) for p in points if p.intersects(p)])
            return coords.astype(int)


def get_indx_poly(vertices,gray_image):
    ################## get the index from the coordinates#####################
    if len(vertices)>3:
        area = coords_in_poly(vertices)
        arr = np.zeros((gray_image.shape))
        coords =  [tuple(area) for area in area.tolist()]
        idx = np.ravel_multi_index(tuple(zip(*coords)), arr.shape)
        arr.flat[idx] = 1
        if arr is None:
            return np.array([0])
        else:
            return arr.astype(int)


def get_masked_image(vertices,gray_image):
    poly_image = np.zeros(gray_image.shape)
    ind = get_indx_poly(vertices,gray_image).astype(int)
    poly_image = 1 - ind
    masked_image = np.ma.masked_where(poly_image, gray_image)
    return masked_image


def generateCMAP(number):
    ##################### to assemble different color set from ########################
    if number<1:
            raise ValueError('number of colors Sz must be at least 1, but is: ',Sz)
    newcol = np.array([], dtype=np.int64).reshape(0,4)
    # color_set_list = ['gist_ncar','tab20','hsv','cividis','jet','Set2','Dark2','rainbow']
    color_set_list = ['gist_ncar','rainbow']
    times = int(number/len(color_set_list*5)+1)
    for color_set in color_set_list*times:
        if color_set == 'tab20':
            produce = 20
        elif color_set == 'gist_ncar' or color_set == 'hsv':
            produce = 10
        else:
            produce = 5
        # cmp = plt.cm.get_cmap(color_set,produce)
        cmp = matplotlib.colormaps[color_set]
        fillColor = np.array([1,1,1,1])
        idx = np.linspace(0, 1, produce)
        newcol = np.concatenate((newcol, cmp(idx)),axis = 0)

        if len(newcol) > number:
            break

    newcmp=ListedColormap(np.array(newcol))
    return newcmp


def apply_color(gray_image,label_img,color_map,transparency=0.5):
    ### this works with any dimension of gray images and apply color based on the labeled image #####
    dim = len(gray_image.shape)
    colored_img = color_map(label_img.astype(int))[...,:3]
    label_mask = np.expand_dims((label_img>0).astype('float'),axis=dim)*transparency    
    gray_image_3ch = np.repeat(np.expand_dims(gray_image,axis=dim),3,axis=dim).astype('float')/np.max(gray_image)
    overlay_img = (1-label_mask)*gray_image_3ch + label_mask*colored_img
    return overlay_img


def inbound(x,x_max):
    x = min(x,x_max)
    x = max(0,x)  
    return x    

#######################################################################################################

class GUI_SEG(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        

        self.pack()
        ################## initialization ##################
        [self.frame, self.xdata, self.ydata, self.zoom_level] = [0, 0, 0, 1]
        [self.x_min, self.x_max, self.y_min, self.y_max] = [0, 0, 0, 1]
        self.clear = False 
        self.neuron_ID = 1
        self.vertices = []
        self.area = []
        self.masked_image = []
        self.masked_label = []
        self.color_map = generateCMAP(max(1,np.max(3000)))
        
        self.file_path = []
        self.selection = False
        self.filter = False
        self.diagnosis = {}

        
  

        
        self.file_path = "data.h5"

        with h5py.File(self.file_path,'r') as f:

            input_image = np.array(f['data'])[0,0]
            self.img_array_original = np.array(f['data'])
            if "label" in list(f.keys()):
                self.label_image_original = np.array(f['label'])   
            else:
                self.label_image_original = np.zeros(self.img_array_original.shape)

        f.close()

        input_gpu = cle.push(input_image)
        self.img_array = np.array(input_gpu)    
        
       
        


        
        self.image = self.img_array[0]
        self.label_image = np.zeros(self.img_array.shape)
        # self.label_color = apply_color(self.img_array,self.label_image,self.color_map,transparency=0.5)
        # self.label_color = self.apply_color_all_layer()
        self.label_color = apply_color(self.img_array,self.label_image,self.color_map,transparency=0.5)
        self.n_frames = int(self.img_array.shape[0])
        
        # create figure and axis
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, dpi=500)
        self.current_frame = self.label_color[0]
        # self.image.seek(self.current_frame)
        self.ax1.imshow(self.current_frame, vmax = np.max(self.current_frame) )
        self.ax1.margins(0) 
        self.ax1.axis('off')   

        self.ax2.imshow(self.current_frame, vmax = np.max(self.current_frame) )
        self.ax2.margins(0) 
        self.ax2.axis('off')

        
        
        # create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        # self.fig.subplots_adjust(top=1, bottom=0.95, left=0.95, right=1, hspace=0.5)
        self.fig.subplots_adjust(top=0.98, bottom=0.01, left=0.01, right=0.98, hspace=-1)
        self.n_frames = int(self.img_array.shape[0])
        
        #### comment the following if not in the test########
        self.create_slider()
        
        self.bind_shortcut()
        
        


        

        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)
        self.create_label_button()
        with h5py.File(self.file_path,'r') as p:
                input_image = np.array(p['data'])
                n_list = input_image.shape[0:-3]              
        p.close()
        self.channel = np.zeros(len(n_list))
        self.master.mainloop()

        
        ########### initilization ################


    def bind_shortcut(self):

        self.canvas.mpl_connect("button_press_event", self.onclick)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("key_press_event", self.handle_keypress)
        

    def handle_keypress(self, event):

        if event.key == "shift":
            self.save_label()   
        elif event.key == "ctrl+z":
            self.redo_label()
        elif event.key == "ctrl+s":
            self.save_file()
        elif event.key == "n":
            self.get_new_ID()
        elif event.key == "0":
            self.redo_ID()
        elif event.key == "backspace":
            self.delete_neuron()
        elif event.key == "super+meta":
            self.select_neuron()
        elif event.key == "ctrl+q":
            self.master.quit
        elif event.key == "Command+f":
            self.look_up()
        elif event.key == "Command+o":
            self.filter_off()
        elif event.key == "Command+d":
            self.run_diagnosis()

        elif event.key == "ctrl+1":
            self.on_channel_select(1, 1)
        elif event.key == "ctrl+2":
            self.on_channel_select(1, 2)
        elif event.key == "ctrl+3":
            self.on_channel_select(1, 3)
        elif event.key == "ctrl+4":
            self.on_channel_select(1, 4)
        elif event.key == "ctrl+5":
            self.on_channel_select(1, 5)
        elif event.key == "ctrl+6":
            self.on_channel_select(1, 6)
                

            
    def create_label_button(self):
        self.filemenu = tk.Menu(self.menu)
        self.filemenu.add_command(label="Open File", command=self.load_file)
        self.filemenu.add_command(label="Save File", accelerator=f"Ctrl+s", command=self.save_file)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", accelerator=f"Ctrl+q", command=self.master.quit)
        self.menu.add_cascade(label="File", menu=self.filemenu)

        with h5py.File(self.file_path,'r') as k:
                input_image = np.array(k['data'])
                n_list = input_image.shape[0:-3]              
        k.close()

        
        self.channel_menus = []
        for i in range(len(n_list)):
            channel_menu = tk.Menu(self.menu)
            self.channel_menus.append(channel_menu)
        

        for i, channel_menu in enumerate(self.channel_menus):
            for j in range(1, n_list[i]+1):
                if i == 1:
                    channel_menu.add_command(label=str(j), accelerator=f"Ctrl+{j}", command=lambda i=i, j=j: self.on_channel_select(i, j))
                else:
                    channel_menu.add_command(label=str(j), command=lambda i=i, j=j: self.on_channel_select(i, j))
                

            self.menu.add_cascade(label=f"Channel {i+1}", menu=channel_menu)        
  
        self.channel = np.zeros(len(n_list))




        self.overlaymenu = tk.Menu(self.menu)
        self.overlaymenu.add_command(label="Overlay Color", accelerator=f"Shift" , command=self.save_label())
        self.overlaymenu.add_command(label="Clear Window", accelerator=f"Ctrl+z" , command=self.redo_label())
        self.overlaymenu.add_separator()
        self.overlaymenu.add_command(label="Get new color", accelerator=f"n", command=self.get_new_ID())
        self.overlaymenu.add_command(label="Eraser", accelerator=f"{0}", command=self.redo_ID())
        self.overlaymenu.add_command(label="Select Neuron", accelerator=f"Command", command=self.select_neuron)
        self.overlaymenu.add_command(label="Delete Neuron", accelerator=f"Delete", command=self.delete_neuron)
        self.overlaymenu.add_separator()
        self.overlaymenu.add_command(label="Look Up Neuron ID", accelerator=f"Command+f", command=self.look_up)
        self.overlaymenu.add_command(label="Filter off", accelerator=f"Command+o", command=self.filter_off)
        self.overlaymenu.add_command(label="Run Diagnosis", accelerator=f"Command+d", command=self.run_diagnosis)
        self.overlaymenu.add_separator()
        self.menu.add_cascade(label="Overlay", menu=self.overlaymenu)


        self.maskmenu = tk.Menu(self.menu)
        self.maskmenu.add_command(label="Load Image",  command=self.load_image)
        self.maskmenu.add_command(label="Load From Label",  command=self.load_from_label)
        self.maskmenu.add_command(label="Save Mask",  command=self.save_mask)
        self.maskmenu.add_command(label="Load Mask",  command=self.load_mask)
        self.menu.add_cascade(label="Create Mask", menu=self.maskmenu)
  

        # self.viewmenu = tk.Menu(self.menu)
        # self.viewmenu.add_command(label="Next Slide", accelerator=f'<Right>' , command=self.next_page())
        # self.viewmenu.add_command(label="Previous Slide", accelerator=f'<Left>' , command=self.prev_page())
        # self.menu.add_cascade(label="Slider", menu=self.viewmenu)

        self.master.config(menu=self.menu)


    def on_channel_select(self, channel_index, selected_index):
        self.channel[int(channel_index)] = int(selected_index-1)
        self.channel = self.channel.astype(int)
        self.show_channel12()

    





    def show_channel12(self):

        self.img_array = np.array(np.array(self.img_array_original)[tuple((self.channel).astype(int))])
        self.label_image = np.array(self.label_image_original[tuple((self.channel).astype(int))])   
        self.label_color = apply_color(self.img_array,self.label_image,self.color_map,transparency=0.5)
        self.n_frames = int(self.img_array.shape[0])
        self.update_frame(self.frame)       
        # self.get_new_ID()


  


        
    def load_file(self):
        '''
        here we only load the h5 file.
        '''
        self.file_path = filedialog.askopenfilename()

        
        if os.path.exists(self.file_path):
            with h5py.File(self.file_path,'r') as l:
                input_image = np.array(l['data'])


                self.img_array_original = np.array(l['data'])
                if "label" in list(l.keys()):
                    self.label_image_original = np.array(l['label'])   
                else:
                    self.label_image_original = np.zeros(self.img_array_original.shape)


                n_list = input_image.shape[0:-3]              
            l.close()
            self.label_image = copy.deepcopy(self.label_image_original)
            # print("test: ",np.unique(self.label_image_original[0]))
            messagebox.showinfo("Notification", "load h5 file from "+self.file_path)
            self.channel = np.zeros(len(n_list))
            self.get_new_ID()
            self.show_channel12()
            self.menu = tk.Menu(self.master)
            self.master.config(menu=self.menu)
            self.create_label_button()
            self.create_slider()
            self.master.mainloop()

            
            # self.create_slider()



    



    def apply_color_all_layer(self):
        self.label_color = apply_color(self.img_array,self.label_image,self.color_map,transparency=0.5)
        for i in range(self.label_color.shape[0]):
            self.label_color[i] = apply_color(self.img_array[i],self.label_image[i],self.color_map,transparency=0.5)
        return self.label_color

    
    def update_frame(self,slice):
        self.frame = int(slice)
        self.zoom_coordinates()
        self.update_original_image()
        self.update_zoom_image()
        self.canvas.draw()
        
  
        
        
    def create_slider(self):
        self.slider = tk.Scale(self, from_=0, to=self.n_frames-1, 
                                # orient=tk.HORIZONTAL, 
                                orient=tk.VERTICAL, 
                                length=600,
                               command=lambda value: self.update_frame(value))
        # self.slider_value = 0  # initialize slider_value attribute
        # self.slider.set(self.slider_value)
        self.frame = int(float(self.slider.get()))
        self.slider.pack()
        self.slider.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.canvas.get_tk_widget().pack()


    



    # def next_page(self):
    #     # update the slider value to move to the next page
    #     self.frame = min(self.n_frames-1, self.frame+1)
    #     self.slider.set(self.frame)
    #     self.update_frame(self.frame)

    # def prev_page(self):
    #     # update the slider value to move to the previous page
    #     self.frame = max(0, self.frame-1)
    #     self.slider.set(self.frame)
    #     self.update_frame(self.frame)



      
    
    def onclick(self,event):
        if event.inaxes == self.ax1:
            x = event.x
            y = event.y
            self.xdata, self.ydata = self.ax1.transData.inverted().transform((x, y))

            self.zoom_coordinates()
            self.update_original_image()
            self.update_zoom_image()
            self.canvas.draw()
    
    
    def on_scroll(self, event):
        # Get scroll value from event on the subplot 2 (self.ax2)
        if event.inaxes == self.ax2:
            # if event.step > 0:
            #     self.zoom_level += 0.1
            # else:
            #     self.zoom_level += -0.1
                
            if event.step<0:           
                if self.zoom_level <= 2 and self.zoom_level> 0:
                    self.zoom_level = self.zoom_level / 2
                elif self.zoom_level > 2:
                    self.zoom_level = self.zoom_level - 1   
                
            elif event.step >0:
                if self.zoom_level <= 1 and self.zoom_level > 0:
                    self.zoom_level = self.zoom_level * 2
                elif self.zoom_level > 1:
                    self.zoom_level = self.zoom_level + 1   
                    
                    
                
            self.zoom_level = np.max([0.1, self.zoom_level])    
            self.zoom_coordinates()
            self.label_color[int(self.frame)] = apply_color(self.img_array[int(self.frame)],self.label_image[int(self.frame)],
                                                            self.color_map,transparency=0.5)
            self.update_original_image()
            self.update_zoom_image()
            self.canvas.draw()
        
        
    def zoom_coordinates(self):
        # [image_width, image_height] = (self.label_color[int(self.frame)]).shape[0:2]
        [image_height, image_width] = (self.label_color[int(self.frame)]).shape[0:2]
        self.x_min = np.max([0,int(self.xdata - image_width / 2 / self.zoom_level)])
        self.x_max = np.min([image_width,int(self.xdata + image_width / 2 / self.zoom_level)])
        self.y_min = np.max([0,int(self.ydata - image_height / 2 / self.zoom_level)])
        self.y_max = np.min([image_height, int(self.ydata + image_height / 2 / self.zoom_level)])
 

    
    def update_zoom_image(self):
 
        self.ax2.clear()
        self.ax2.axis('off')

        if self.zoom_level == 1:
            self.ax2.imshow(self.label_color[int(self.frame)],vmax = np.max(self.label_color[int(self.frame)]))

            
        elif self.zoom_level != 1 and  self.x_min != self.x_max and  self.y_min != self.y_max:
            # if not self.vertices:
            #     self.ax2.imshow(self.img_array[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max], 'gray')
            # else:
            # img = self.label_color[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max]
            if self.filter:
                self.label_image[self.label_image!=self.neuron_ID] = 0

            zoom_image = apply_color(self.img_array[int(self.frame), self.y_min:self.y_max, self.x_min:self.x_max],
                                    self.label_image[int(self.frame), self.y_min:self.y_max,self.x_min:self.x_max],
                                    self.color_map, transparency = 0.5)

            self.ax2.imshow(zoom_image, vmax = np.max(zoom_image))
        # self.get_new_ID()
        self.ax2.margins(0) 
        self.ax2.set_title("current labeling neuron ID "+str(int(self.neuron_ID)), fontsize = 1, y = 0.91, fontweight='bold')
            

    
    
    def update_original_image(self):
        
        '''
        Adding the bounding box on the original image to show where the zoomed image lie in
        '''
        self.ax1.clear()
        self.ax1.axis('off')
        if self.filter:
            self.label_image[self.label_image!=self.neuron_ID] = 0

        self.label_color[int(self.frame)] = apply_color(self.img_array[int(self.frame)],self.label_image[int(self.frame)],
                                                            self.color_map,transparency=0.5)
        self.ax1.imshow(self.label_color[int(self.frame)],vmax = np.max(self.label_color[int(self.frame)]))
        self.ax1.set_title("Original Image In Channel "+str(tuple((self.channel+1).astype(int))), fontsize = 1, y = 0.91, fontweight='bold')
        self.ax1.margins(0) 


        if self.zoom_level != 1 and  self.x_min!=self.x_max and  self.y_min!=self.y_max:
            self.bbox = Rectangle((self.x_min, self.y_min), (self.x_max-self.x_min), (self.y_max-self.y_min), 
                fill=False, edgecolor="yellow", linewidth=0.1, alpha = 1)
            self.ax1.add_patch(self.bbox)

        # self.canvas.draw()
        
        

                
    ################################# To draw a polygon on the subplot2 #################################           
    def on_press(self, event):
        # Start new polygon if left mouse button pressed and initialize the polygon vertices
        self.vertices = []
        if event.inaxes == self.ax2 and event.button == 1  and self.zoom_level!=1:
            self.vertices.append((event.x, event.y))
            # x, y = self.ax2.transData.inverted().transform((event.xdata, event.ydata))
            # self.vertices.append((x, y))
       
    def on_motion(self, event):
        # Update polygon if left mouse button pressed and mouse moved
        if event.button == 1 and len(self.vertices) >= 1:   
            if event.xdata is not None and event.ydata is not None:   
                self.vertices[-1] = (event.xdata, event.ydata)
                self.vertices.append((event.xdata, event.ydata))
                self.ax2.plot(np.array(self.vertices)[:,0], np.array(self.vertices)[:,1], 'r', linewidth = 0.1)
                self.ax2.margins(0) 
                self.canvas.draw()   
            else:
                self.vertices = []



    def on_release(self, event):
        # End polygon if left mouse button released
        if event.button == 1 and len(self.vertices) >= 3:
            self.vertices.append(self.vertices[0])           
            # cmap = plt.cm.get_cmap('RdBu_r')
            cmap = matplotlib.colormaps['RdBu_r']
            gray_image = self.img_array[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max]   
            self.area = (get_indx_poly(self.vertices,gray_image))
            # if self.area is not None:
            if  len(self.area)>1:
                # self.masked_image = get_masked_image(self.vertices, gray_image)
                self.ax2.clear()
                zoom_image = self.label_color[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max]

                if self.clear:
                    self.label_color_zoom = apply_color(gray_image,self.area * 0,self.color_map,transparency=0.5)
                else:
                    self.label_color_zoom = apply_color(gray_image,self.area * self.neuron_ID,self.color_map,transparency=0.5)


                self.ax2.imshow(self.label_color_zoom, vmax = np.max(self.label_color_zoom))
                self.ax2.axis('off')
                self.ax2.margins(0) 
                self.canvas.draw()
                self.vertices = []
        

            

            
    
    def save_label(self):        
        gray_image = self.img_array[int(self.frame)]          
        # neuron_ID = 1
        if  self.area is not None:
            if len(self.area)>1:
                ind = (self.area) > 0
                if self.clear:
                    self.label_image[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max][ind]  = 0
                    self.label_image_original[tuple((self.channel).astype(int))][int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max][ind]  = 0

                else:
                    self.label_image[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max][ind]  = self.neuron_ID 
                    self.label_image_original[tuple((self.channel).astype(int))][int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max][ind]  = self.neuron_ID 

                
                # if self.filter:
                #     self.label_image = copy.deepcopy(self.label_image_original[tuple((self.channel).astype(int))])
                #     self.label_image[self.label_image!=self.neuron_ID] = 0
                # else:
                #     self.label_image = self.label_image_original[tuple((self.channel).astype(int))]

                

                self.clear = False
                self.label_color[int(self.frame)] = apply_color(self.img_array[int(self.frame)],self.label_image[int(self.frame)],
                                                                self.color_map,transparency=0.5)


                self.update_original_image()
                self.update_zoom_image()
                self.canvas.draw()
                # self.update_frame()
                self.vertices = []
                self.area = (get_indx_poly(self.vertices,gray_image))


    ################################# To look up the polygon using the neuron_ID ################################# 
    def look_up(self):
        # number = 1
        number = tk.simpledialog.askinteger("Enter Number", "Please enter the neuron ID:")
    
        if number is not None:
            self.neuron_ID = int(number)
            ind = np.where(self.label_image_original == self.neuron_ID)
            if np.array(ind).shape[1] > 0:
                self.channel = np.array(ind)[0:-3,0]
                self.label_image = copy.deepcopy(self.label_image_original[tuple((self.channel).astype(int))])
                self.frame = int((ind[-3])[0])

                ind2 = np.where(self.label_image[int(self.frame)] == self.neuron_ID)
                [self.xdata, self.ydata]= [int(ind2[1][0]),int(ind2[0][0])]
                self.zoom_level = 17
                self.filter = True
                self.slider.set(self.frame) 

                self.zoom_coordinates()
                self.update_original_image()
                self.update_zoom_image()
                self.canvas.draw()
            else:
                messagebox.showinfo("Notification", "The labeled neuron ID " + str(self.neuron_ID) + " doesn't exist in this h5 file")
        else:
            messagebox.showinfo("Notification", "No valid integer is entered!")
        


    def filter_off(self):
        self.filter = False
        self.label_image = self.label_image_original[tuple((self.channel).astype(int))] 
        self.update_original_image()
        self.update_zoom_image()
        self.canvas.draw()



    def run_diagnosis(self):
        diagnosis_obj = NeuronDiagnosis()
        diagnosis_obj.diagnose_neurons(self.label_image_original)   
        self.diagnosis = diagnosis_obj.diagnosis
        messagebox.showinfo("Dictionary", diagnosis_obj.diagnosis)


    ################################# To create mask #################################################################
    def load_image(self):
        '''
        Load the whole c elegans image "color.jpg" and create a mask to train only on these area
        '''
        self.file_path = filedialog.askopenfilename()
        if os.path.exists(self.file_path):
            image = cv2.imread(self.file_path)
            self.img_array = image[None,:,:,0]
            self.frame = 0
            self.label_image = np.zeros(self.img_array.shape)

            # self.create_label_button()
            self.slider.grid_remove()
            self.update_original_image()
            self.update_zoom_image()
            self.canvas.draw()
            self.master.mainloop()

            
    def load_from_label(self):
        '''
        Load the maximum projection all labeled image to create mask and train only these area
        '''
        ind = self.label_image_original > 0
        image = np.sum(ind,axis = tuple(np.arange(len(ind.shape))[0:-2])) > 0
        self.img_array = image[None,:,:]
        self.frame = 0
        self.label_image = np.zeros(self.img_array.shape)
        # self.create_label_button()
        self.slider.grid_remove()
        self.update_original_image()
        self.update_zoom_image()
        self.canvas.draw()
        self.master.mainloop()
    
    
    def save_mask(self):
        
        self.file_path = filedialog.askopenfilename()

        if self.file_path:
            with h5py.File(self.file_path, "a") as g:
       
                if 'mask' in list(g.keys()):
                    del g['mask']
                    dset = g.create_dataset('mask', data = self.label_image[0])
                else:
                    dset = g.create_dataset('mask', data = self.label_image[0])
            g.close()


    def load_mask(self):
        self.file_path = filedialog.askopenfilename()

        if self.file_path:
            with h5py.File(self.file_path, "a") as m:
       
                if 'mask' in list(m.keys()):
                    image = np.array(m['mask'])
                    self.img_array = image[None,:,:]
                    self.label_image = (self.img_array)
                else:
                    messagebox.showinfo("Notification", "There is no 'mask' saved in the file "+self.file_path)
            m.close()


            
            self.frame = 0
            self.slider.grid_remove()
            self.update_original_image()
            self.update_zoom_image()
            self.canvas.draw()
            self.master.mainloop()

    ################################# To assign class #################################################################
    def get_new_ID(self):
        rest = list(set(list(np.array(np.arange(np.max(self.label_image_original)+1)))) - set(list(np.unique(self.label_image_original))))
        if not self.filter:
            if not rest:
                self.neuron_ID = np.max(self.label_image_original) + 1
            else:
                self.neuron_ID = rest[0]
        




        
        
    ################################# To cancel the polygon ################################# 
    def redo_label(self):
        #### "control+z" to clear all the label in the zoomed window 
        self.label_image[int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max] = 0
        self.label_image_original[tuple((self.channel).astype(int))][int(self.frame),self.y_min:self.y_max,self.x_min:self.x_max] = 0
        self.label_color[int(self.frame)] = apply_color(self.img_array[int(self.frame)],self.label_image[int(self.frame)],
                                                        self.color_map,transparency=0.5)
        # self.update_frame()
        self.update_zoom_image()
        self.update_original_image()
        self.canvas.draw()
    

    def redo_ID(self):
        self.clear = "True"


    def delete_neuron(self):
        if self.selection:
            ind = self.label_image_original == self.neuron_ID
            self.label_image_original[ind] = 0
            self.img_array = np.array(np.array(self.img_array_original)[tuple((self.channel).astype(int))])
            self.label_image = np.array(self.label_image_original[tuple((self.channel).astype(int))])   
            self.label_color[int(self.frame)] = apply_color(self.img_array[int(self.frame)],self.label_image[int(self.frame)],
                                                            self.color_map,transparency=0.5)

            self.update_zoom_image()
            self.ax2.set_title("delete function is obtained"+str(int(self.neuron_ID)), fontsize = 1, y = 0.91, fontweight='bold')
            self.ax2.margins(0) 
            self.update_original_image()
            self.canvas.draw()
            self.selection =  False


    def select_neuron(self):
        ## click the neuron and press command to select 
        self.neuron_ID = self.label_image[int(self.frame),int(self.ydata),int(self.xdata)]
        self.update_zoom_image()
        self.ax2.set_title("select neuron ID "+str(int(self.neuron_ID)), fontsize = 1, y = 0.91, fontweight='bold')
        self.ax2.margins(0) 
        self.canvas.draw()
        self.selection = True








    ################################# To save the file ################################# 
    def save_file(self):


        if not os.path.exists(self.file_path):
            self.file_path = filedialog.askopenfilename()


        with h5py.File(self.file_path, "a") as g:
        # with h5py.File(self.file_path,'a')
            if 'label' in list(g.keys()):
                del g['label']
                # g['label'][tuple((self.channel).astype(int))] = self.label_image
                # g['label'] = self.label_image_original
                dset = g.create_dataset('label', data = self.label_image_original)

            else:
                # data = np.zeros(g['data'].shape[-3:None])
                # data = np.zeros(g['data'].shape)
                # data[tuple((self.channel).astype(int))] = self.label_image
                # dset = g.create_dataset('label', data = data)
                dset = g.create_dataset('label', data = self.label_image_original)
        g.close()

        # print("test: ",np.unique(self.label_image_original[0]))

        messagebox.showinfo("Notification", "Save current data as 'label' in the path "+self.file_path)

                           


    
root = tk.Tk()
root.geometry("3300x1500")
root.title("C elegans Neuron Segmentation")
app = GUI_SEG(master=root)
app.mainloop()





