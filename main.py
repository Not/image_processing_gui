"""
Program do rozpoznawania kart do gry ze zdjęcia
"""


import numpy as np
import tkinter 
import tkinter.filedialog 
from PIL import ImageTk,Image
from tkinter import messagebox
from tkinter import ttk
import cv2 as cv
import card_finder_model as mdl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
from concurrent import futures
import functools
import json


thread_pool_executor = futures.ThreadPoolExecutor(max_workers=1)
 
 
def tk_after(target):
 
    @functools.wraps(target)
    def wrapper(self, *args, **kwargs):
        args = (self,) + args
        self.master.after(0, target, *args, **kwargs)
 
    return wrapper
 
 
def submit_to_pool_executor(executor):
    
    def decorator(target):
 
        @functools.wraps(target)
        def wrapper(*args, **kwargs):
            result = executor.submit(target, *args, **kwargs)
            result.add_done_callback(executor_done_call_back)
            return result
 
        return wrapper
 
    return decorator
 
 
def executor_done_call_back(future):
    exception = future.exception()
    if exception:
        raise exception
 
    

class View():
    """Odpowiada za wyswietlanie interfejsu"""
    def __init__(self, master):
        
        
        self.master= master
        master.geometry('820x500')
       # master.iconbitmap('icon.ico')
        #master.iconphoto(False, tkinter.PhotoImage(file='icon.png'))
        
        self.frame_buttons=tkinter.Frame(master,relief=tkinter.GROOVE,borderwidth=3,width=300)
        self.frame_buttons.pack(fill=tkinter.Y,side=tkinter.LEFT,expand=False,padx=5,pady=5)
    
        self.frame_img=tkinter.Frame(master,relief=tkinter.FLAT,borderwidth=2,width=400)
        self.frame_img.pack(fill=tkinter.BOTH,side=tkinter.LEFT,expand=True,pady=5)
    
        vscrollbar = AutoScrollbar(self.master)
        vscrollbar.pack(fill=tkinter.Y,side=tkinter.RIGHT,expand=False)
       
        
        self.frame_result=tkinter.Canvas(master,relief=tkinter.GROOVE,
                                         borderwidth=3,width=210,
                                          yscrollcommand=vscrollbar.set)
        vscrollbar.config(command=self.frame_result.yview)
        self.frame_result.pack(fill=tkinter.Y,side=tkinter.LEFT,padx=5,pady=5)
        
        self.frame_img.rowconfigure(0, weight=1)
        self.frame_img.columnconfigure(0, weight=1)
        
        self.content_frame=tkinter.Frame(self.frame_img,borderwidth=8,
                                relief=tkinter.RAISED,
                                background="blue")
        
        self.tab_frame=tkinter.Frame(self.content_frame, background="blue",height=20)
        self.tab_frame.pack(fill=tkinter.X)
        
        self.canvas = tkinter.Canvas(self.content_frame)
        #----NB
        self.tab_main=ttk.Notebook(self.tab_frame)
        tab_picture=ttk.Frame(self.tab_main)
        tab_thresh=ttk.Frame(self.tab_main)
        self.tab_main.add(tab_picture,text="Picture")
        #tab_main.add(tab_thresh,text="Threshold")
        self.tab_main.pack(expand=True,fill=tkinter.BOTH)
        #----BTN OPEN
        
        self.icon_1=tkinter.PhotoImage(file = r"resources/pic1.png")
        self.btn_open = tkinter.Button(self.frame_buttons, text="  Load picture",
                                       image=self.icon_1,
                                       width=150,
                                       compound = tkinter.LEFT)
        self.btn_open.pack(fill=tkinter.BOTH)
        
        #----LF
        settings_lf = ttk.Labelframe(self.frame_buttons, text='Settings',width=100, height=100)
        settings_lf.pack(fill=tkinter.BOTH)
        self.quality = tkinter.Scale(settings_lf, from_=1, to=10,orient=tkinter.HORIZONTAL)
        self.quality.set(5)
        self.quality.pack(pady=8)
        tkinter.Label(settings_lf,text='<-Speed       Quality->').pack(fill=tkinter.BOTH)
        
        #-----BTN RUN
        self.icon_2=tkinter.PhotoImage(file = r"resources/pic2.png")
        self.btn_run = tkinter.Button(self.frame_buttons, text="  Recognize",
                                      image=self.icon_2,
                                       compound = tkinter.LEFT,
                                       width=150,
                                       state=tkinter.DISABLED)
        self.btn_run.pack(pady=10)
        
        #-----PROGRESS
        self.lbl_progress=tkinter.Label(self.frame_buttons)
        self.progress = tkinter.ttk.Progressbar(self.frame_buttons, orient = tkinter.HORIZONTAL, 
                                                length = 100, mode = 'determinate') 
        
        self.btn_export = tkinter.Button(self.frame_buttons, text="Export JSON",state=tkinter.DISABLED)
        self.btn_export.pack(fill=tkinter.BOTH)
        
        #----CLEAR/EXIT
        frame_exit=tkinter.Frame(self.frame_buttons,)
        frame_exit.pack(fill=tkinter.BOTH,side=tkinter.BOTTOM)
        self.btn_clear=tkinter.Button(frame_exit,text="Clear",width=9)
        self.btn_clear.pack(side=tkinter.LEFT)
        self.btn_exit=tkinter.Button(frame_exit,text="Exit",width=9,command=master.destroy)
        self.btn_exit.pack(side=tkinter.RIGHT)

        #----SCROLLABLE
        frame_cards = tkinter.Frame(self.frame_result)
        self.frame_result.create_window(0, 0,  window=frame_cards)
        frame_cards.update_idletasks()
        
        self.frame_result.config(scrollregion=self.frame_result.bbox("all"))
        self.frame_cards=frame_cards

        #------FIRST PICTURE
        img0=ImageTk.PhotoImage(Image.open(r"resources/load.png"))
        self.img0=img0
        self.printed_image=self.canvas.create_image(0,0,
                                      anchor=tkinter.NW, 
                                      image=img0) 
        self.canvas.pack()
        self.current_image=None
        
       # self.set_aspect(self.content_frame, self.frame_img, 1.0) 
    @tk_after
    def update_picture(self):
        global img 
        
        size=(self.canvas.winfo_width(),self.canvas.winfo_height())
        
        img = ImageTk.PhotoImage(image=Image.fromarray(self.current_image).resize(size,Image.NEAREST)) 
        
        self.canvas.itemconfig(self.printed_image, image=img)
        return
    def show_progress(self,value=0,active=False):
        if(active):
            self.progress['value']=0
    def remove_tabs(self):
        for item in self.tab_main.winfo_children()[1:]:
            item.destroy()
    def remove_cards_thumb(self,event=None):
        for item in self.frame_cards.winfo_children():
            item.destroy()
        if event is not None:
            self.remove_tabs()
    
        
    
    def set_aspect(self,content_frame, pad_frame, aspect_ratio):
        #Funkcja zapewniająca zachowanie stałych proporcji widżetu 
    
        def enforce_aspect_ratio(event):
            # Funkcja wylicza rozmiary widżetu
    
            # start by using the width as the controlling dimension
            desired_width = event.width
            desired_height = int(event.width / aspect_ratio)
            
            # if the window is too tall to fit, use the height as
            # the controlling dimension
            if desired_height > event.height:
                desired_height = event.height
                desired_width = int(event.height * aspect_ratio)
    
            # place the window, giving it an explicit size
            content_frame.place(in_=pad_frame,
                                x=int((pad_frame.winfo_width()-desired_width)/2), y=0, 
                                width=desired_width, height=desired_height)
            self.canvas.config(width=desired_width, height=desired_height)
            self.canvas.update()
           
            if(self.current_image is not None):
                #print("not none")
                self.update_picture()
      
        pad_frame.bind("<Configure>", enforce_aspect_ratio)
        
   
class AutoScrollbar(tkinter.Scrollbar):
    # TODO, na razie nie dziala
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            
            #print("rem")
            #self.pack_forget()
            pass
        else:
            
            #print("add")
            self.pack()
        tkinter.Scrollbar.set(self, lo, hi)


        
class Controller():
    
    def __init__(self):
        self.root = tkinter.Tk() #ok, jakis glowny TK
        self.model=mdl.cf() #model  liczenie obrazka
        self.view=View(self.root) #widok 
        self.view.btn_open.bind("<Button>",self.open_file)
        self.view.btn_run.bind("<Button>",self.main)
        self.view.btn_export.bind("<Button>",self.export)
        self.view.btn_clear.bind("<Button>",self.view.remove_cards_thumb)
        self.view.tab_main.bind("<<NotebookTabChanged>>",self.tab_selected)
        self.view.set_aspect(self.view.content_frame, self.view.frame_img, 3/4)
        
        self.file_loaded=False
        self.in_progress=False
        
        self.img_array=None
        self.thresh1=None
        self.filtered_objects_mask=None
        self.result={}
        self.cards_views={}
  
  
    def run(self):
        self.root.title("Rozpoznawanie Kart")
        self.root.deiconify()
        self.root.mainloop()
    
    def export(self,event):

        filepath= tkinter.filedialog.asksaveasfilename(filetypes = (("txt files",".txt"),
                                                                    ("JSON files",".json"),
                                                                    ("all files","*.*")))
        print(filepath) 
        if not filepath:
            messagebox.showinfo("Information","Operacja anulowana") 
            return
        
        with open(filepath, 'w') as outfile:
             outfile.write(json.dumps(self.result,indent=2))
             
    def show_card_thumb(self,card):
        
        cards_thumb= cv.cvtColor(cv.imread(r"resources/deck.png",cv.IMREAD_UNCHANGED),cv.COLOR_BGR2RGBA)
        
        frame_cardinfo=tkinter.Frame(self.view.frame_cards, background="blue",
                                     relief=tkinter.RAISED,borderwidth=3,
                                     height=100,width=180)
        frame_cardinfo.pack(padx=2,pady=3)
       
        labels=["figura","kolor","Położenie","rozmiar","kąt"]

        
        values=[card["value"],card["suit"],
                card["center"],card["size"],round(card["angle"],2)]
        d=dict(zip(labels, values))
        for i,(label,value) in enumerate(d.items()):
            tkinter.Label(frame_cardinfo, text = label,justify=tkinter.LEFT,
                          anchor="w",relief=tkinter.SUNKEN).grid(row=i,
                                                                 column=1,
                                                                 sticky = "nsew",
                                                                 pady = 1)
            tkinter.Label(frame_cardinfo, text = value,
                          relief=tkinter.SUNKEN).grid(row=i,
                                                      column=2,
                                                      sticky = "nsew", 
                                                      pady = 1)
            
        frame_cardinfo.columnconfigure(0, weight=1, minsize=60)
        frame_cardinfo.columnconfigure(1, weight=1, minsize=50)
        frame_cardinfo.columnconfigure(2, weight=1, minsize=60)
        #frame_cardinfo.rowconfigure(list(range(len(labels))), weight=1, minsize=10)


        #global img2
        w,h=cards_thumb.shape[1]/13,cards_thumb.shape[0]/4
        x0,y0=int(w*card["value"][1]-w),int(h*card["suit"][1])
        w,h=int(w),int(h)
        
        img2 = ImageTk.PhotoImage(image=Image.fromarray(cards_thumb[y0:y0+h,x0:x0+w])) 
        
        lbl=tkinter.Label(frame_cardinfo, text = label,image=img2,background="blue")
        lbl.grid(row=0,column=0,rowspan = 5,sticky = "nsew",pady=2,padx=2)
        lbl.image=img2
        self.view.frame_cards.update_idletasks()
        self.view.frame_result.config(scrollregion=self.view.frame_result.bbox("all"))
         
   
  
    def open_file(self,event): #tutaj powinno się wybrac plik do cf

        self.view.progress.forget()
        self.view.lbl_progress.forget()
        filepath= tkinter.filedialog.askopenfilename(filetypes=[("Picture", "*.jpg"),
                                              ("All Files", "*.*")])
        if not filepath:
            messagebox.showinfo("Information","Operacja anulowana") 
            return

        self.img_array=cv.cvtColor( cv.imread(filepath),cv.COLOR_BGR2RGB)
        
        ratio=self.img_array.shape[1]/self.img_array.shape[0]
        self.view.current_image=self.img_array
        
        self.view.set_aspect(self.view.content_frame, self.view.frame_img, ratio)
        self.view.update_picture()
        
        print(f"loaded file: {filepath}") 
        self.file_loaded=True;
        self.view.btn_run.config(state="normal")
        self.view.remove_tabs()
        self.view.remove_cards_thumb()
        
   
    def tab_selected(self,event):
        tab = event.widget.index("current")
        pictures=[self.img_array, self.thresh1, self.filtered_objects_mask]
        if(self.img_array is not None and tab<=2):
            self.view.current_image=pictures[tab]
            self.view.update_picture()
        if(tab>2):
            self.view.current_image=self.cards_views[str(tab-3)]
            self.view.update_picture()
        
    @submit_to_pool_executor(thread_pool_executor)    
    def main(self,event):
        
        
        self.view.remove_tabs()
        if(not self.file_loaded or self.in_progress):
            return
        self.view.lbl_progress.pack();
        self.view.progress.pack()
        self.in_progress=True;
        self.view.btn_open.configure(state=tkinter.DISABLED)
        self.view.btn_export.config(state=tkinter.DISABLED)
        #self.view.show_progress()
        #-----------------------------------------------------------SKALOWANIE
        upper=100000
        org_size=self.img_array.size/3
        quality = self.view.quality.get()
        scale=math.sqrt(org_size/(quality*(upper/10)))
        newX,newY = self.img_array.shape[1]/scale, self.img_array.shape[0]/scale
        img_scaled = cv.cvtColor(cv.resize(self.img_array,(int(newX),int(newY))),cv.COLOR_RGB2GRAY)
        img_full_color=self.img_array.copy()
        self.model.show(img_scaled, "Obraz przeskalowany")
        #-----------------------------------------------------------BINARYZACJA
        self.view.lbl_progress.config(text="Thresholding")
        self.otsu=self.model.find_thresh_otsu(img_scaled)
        self.thresh1=self.model.thresh(img_scaled,self.otsu*1.0)*255
        tab_binary=ttk.Frame(self.view.tab_main)
        self.view.tab_main.add(tab_binary,text="Binary")
        self.view.tab_main.select(tab_binary)
        self.view.progress['value']=5
        self.model.show(self.thresh1, "Obraz zbinaryzowany (Otsu)")
     
        #----------------------------------------------------------SEGMANTACJA
        self.view.lbl_progress.config(text="Image segmentation")
        background=self.model.select_object(self.thresh1,(5,150),None) 
        objects_mask=np.logical_not(background["mask_img"]).astype(np.uint8) 
        self.view.progress['value']=10
        self.model.show(objects_mask, " Potencjalne Obiekty")
        #--------------------------------------------------FILTROWANIE OBIEKTÓW
        self.view.lbl_progress.config(text="Object filtering")
        size_factor =0.020 #Rozpoznawane będą obiekty powyżej 3.5% obrazu
        size_pix=self.thresh1.size*size_factor
        cards_list=self.model.filter_object(objects_mask,size_pix) 
        
           
        self.filtered_objects_mask=sum(c["mask_img"] for c in cards_list)*255
        for  i,card in enumerate(cards_list): #TYLKO DO ZAZNACZENIA KART
            ctr=tuple(map(int,self.model.center(card["mask_img"])))
            card["center"]=ctr
            card["index"]=i
            cv.circle( self.filtered_objects_mask,ctr,3,(0,0,0))
        tab_objects=ttk.Frame(self.view.tab_main)
        self.view.tab_main.add(tab_objects,text="Objects")
        self.view.tab_main.select(tab_objects)
        self.model.show(self.filtered_objects_mask, "Obiekty")
        self.view.progress['value']=15
        self.view.lbl_progress.config(text=f"Found {len(cards_list)} cards")
        
        #---------------------------------------------------------------------
        for index,card in enumerate(cards_list):
            card["probability"]=100 
            tab_card=ttk.Frame(self.view.tab_main)
            self.cards_views[str(index)]=card["mask_img"]*255
            self.view.tab_main.add(tab_card,text=f"card {index}")
            #self.view.tab_main.select(tab_card)
            
            self.view.lbl_progress.config(text=f"Finding edges-card{index}")
            self.model.show(card["mask_img"],f'maska karty {index}')
            
            #-------------------------------------------------WYKRYCIE KRAWEDZI
            kernel=np.array([[1,1,1], 
                             [1,-8,1],
                             [1,1,1]])
                
            card_border1=self.model.operation(card["mask_img"],kernel,"convolution")  
            card_border1=self.model.thresh(abs(card_border1))
            self.view.progress['value']=15+(75/len(cards_list))*(index+0.2)

            self.model.show(card_border1,f"Krawędź karty")
            #self.cards_views[str(index)]=card_border1*255
            #self.view.tab_main.select(tab_card)
            #self.view.update_picture()
            
             #---------------------------------------------WYKRYCIE BOKÓW KARTY
            self.view.lbl_progress.config(text=f"Finding lines-card{index}")
            lines=self.model.line_transform(card_border1,(400,400)) ##TRANSFROMACJA
            self.model.show(lines,"Hough")
                #-----------------------
            points=self.model.get_lines(lines,4,115,card_border1.shape)
            
            self.view.progress['value']=15+(75/len(cards_list))*(index+0.4)
            
            #----------------------------------------------WYKRYCIE NAROŻNIKÓW
            self.view.lbl_progress.config(text=f"Finding corners-card{index}")
            corners=self.model.find_corners(points["ab"]) 
            #----------------------------------------------KONTROLA NAROŻNIKI
            if len(corners) != 4:
                print("This is not a card")
                #cards_list.remove(card)
                card["probability"]=0
                self.view.tab_main.hide(self.view.tab_main.winfo_children()[index+3])
                continue
            corners=self.model.sort_corners(corners)
            #--------------------------------------------KONTROLA POLE
            delta=0.1 #10%
            print(f"!!!!maska: {card['size']}, pole: {self.model.polygon_area(corners)}")
            pol_area=self.model.polygon_area(corners)
            if((abs(pol_area-card['size'])/card['size'])>delta):
                    print("area error")
                    card["probability"]=0
                    self.view.tab_main.hide(self.view.tab_main.winfo_children()[index+3])
                    continue
            self.view.progress['value']=15+(75/len(cards_list))*(index+0.6)
            
            #------------------------------------ZAPISANIE INFORMACJI O KARCIE
            card["corners"]=corners
            card["edges"]=self.model.sort_lines(corners)
            card["angle"]=math.degrees(math.atan(2/(card["edges"][0][0]+
                                                    card["edges"][1][0])))
             
            #-----------------------------------------------------WIZUALIZACJA 
            img_draw=self.img_array.copy()  
            scaleY=self.img_array.shape[0]/img_scaled.shape[0]
            scaleX=self.img_array.shape[1]/img_scaled.shape[1]
            #print(f"org:{scale} ,y {scaleY}, x{scaleX}")
            thickness=np.clip( int(img_draw.size/1000000),1,20)
            
                 #------------------------TRANSFORMATA  
            for point_ar in points["ar"]: #WIZUALIZACJA PROSTYCH W TRANSFORMACIE
                cv.circle(lines, tuple(map(int,point_ar)),4, (255,0,0))
            self.model.show(lines,"Wykryte proste")
                 #------------------------PROSTE   
            for point_ab in points["ab"]: 
                a,b=point_ab
                h,w,z=self.img_array.shape
                cv.line(img_draw,(0,int(b*scale)),(int(w),int(a*w+b*scale)),(200,200,200),
                        thickness=thickness)
                 #-------------------------RAMKA
           
            print(f"scale{thickness}")
            for i in range(4):
                p1=(int(corners[i][0]*scale),int(corners[i][1]*scale))
                p2=(int(corners[(i+1)%4][0]*scale),int(corners[(i+1)%4][1]*scale))
                cv.line(img_draw,p1,p2,(255,0,0),6*thickness)
                      #-------------------------NAROŻNIKI
            for i, p in enumerate(corners):

                cv.circle(img_draw, (int(p[0]*scaleX),int(p[1]*scaleY)),3*thickness,
                            (0,255,0),thickness=4*thickness)
            
            
                
            self.model.show(img_draw,"Wykryte boki karty")
            
            self.cards_views[str(index)]=img_draw
            self.view.tab_main.select(tab_binary)
            self.view.tab_main.select(tab_card)
            

            width,height=130,200 
            warped=np.zeros((height,width))
            rectangle=np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)
            source=np.array(card['corners'],dtype=np.float32)*scale #narozniki na glownym obrazie
            
            matrix=cv.getPerspectiveTransform(source, rectangle)
            img_full=cv.cvtColor(img_full_color,cv.COLOR_BGR2GRAY)
            warped=cv.warpPerspective(img_full, matrix, (width,height))
            self.model.show(warped,"karta bez perspetywy")
            
            self.view.progress['value']=15+(75/len(cards_list))*(index+0.8)
            self.view.lbl_progress.config(text=f"Finding symbols-card{index}")
            border=20
            roi=warped[border:-border,border:-border]
            #median=np.median(warped)

            roi_t=self.model.thresh(roi,self.model.find_thresh_otsu(warped)) #binaryzacja jednej karty
            roi_t=self.model.operation(roi_t,self.model.get_circle(2),"median")#flitr medianowy
            print(f"sum {roi_t.sum()}")
            self.model.show(roi_t,"Tu szukam symboli")
            
                
            #-----------------rozpoznanie symboli
            patterns=["pik","kier","karo","trefl"]
            corr=dict.fromkeys(patterns,0)
            
            for pattern in patterns:
                
                img=np.logical_not(cv.imread(r"resources/"+pattern+".png",0))*2-1
                card_img=np.logical_not(roi_t)
                result=self.model.operation(card_img,img,'convolution')
                result_flip=self.model.operation(card_img,np.flip(img),'convolution')#ZBADANIE KORELACJI
                self.model.show(result, f"Porownanie z {pattern}")
                corr[pattern]=max(result.max(),result_flip.max())
            type_of_card=max(corr.keys(),key=lambda x:corr[x])    
            card['suit']=(type_of_card,patterns.index(type_of_card))#kolor,wartosc
            #----------------------------
            figury=["As","Dwa","Trzy","Cztery","Pięć","Szesć","Siedem","Osiem",
                "Dziewięć","Dziesięć","Figura","Figura","Figura"]
            if(roi_t.sum()>8500):
                print("to nie figura")
                
                
                #-------------------segmentacja symboli
                size_factor =0.02 #Rozpoznawane będą obiekty powyżej 3.5% obrazu
                size_pix=roi.size*size_factor
                symbols=self.model.filter_object(np.logical_not(roi_t).astype(np.uint8), size_pix) #SEGMEMTACJA SYMBOLI 
                self.view.progress['value']=15+(75/len(cards_list))*(index+0.9)
                self.view.lbl_progress.config(text=f"Recognizing-card{index}")
                #------------------srodki symboli
                for symbol in symbols:
                   symbol["center"]=self.model.center(symbol["mask_img"]) 
                #----------------- 
    
                card['value']=(figury[len(symbols)-1],len(symbols)) #figura, wartosc
                
                matrix_inv=np.linalg.inv(matrix)
                for symbol in symbols:
                    center=np.dot(matrix_inv,np.append((symbol['center']+np.array([border,border])),1))
                    cv.circle(img_draw,tuple(map(int,center[:2]/center[2])),6*thickness,(0,0,255),thickness=thickness)
                # cv.circle(warped,tuple(map(int,symbol['center']+np.array([border,border]))),1,255,thickness=2)
                self.model.show(warped,"wynik")
                self.cards_views[str(index)]=img_draw
                self.model.show(img_draw, "Znalezione symbole")
                self.view.tab_main.select(tab_binary)
                self.view.tab_main.select(tab_card)
                
                
            
            else:
                card['value']=("Figura",13)
            print(f"Ta karta to { card['value'][0]} {type_of_card}")
            self.show_card_thumb(card)
    
        self.view.lbl_progress.config(text="Done!")
        self.view.progress['value']=100
        
        self.result=[d for d in cards_list if d.get('probability')>80]
        for r in  self.result:
            
            r.pop('mask_img')
            r['edges']=r['edges'].tolist()
        self.in_progress=False
        self.view.btn_export.config(state="normal")
        self.view.btn_open.config(state="normal")
        
        
  
 
 
if __name__ == '__main__':
    
    c = Controller()
    c.run()


