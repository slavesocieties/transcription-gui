import tkinter as tk
import customtkinter as ctk
import requests
import os
from PIL import Image
import shutil
from segment import *

class TrGUI:
    def __init__(self, root):
        self.curr_segment = None
        self.vol_id = None
        self.img_id = None
        self.root = root
        self.root.geometry('720x270')
        self.root.title('SSDA transcription tool')
        self.user_input = tk.StringVar()  # Instance attribute to store input

        self.title = ctk.CTkLabel(self.root, text="Enter an SSDA volume ID to transcribe from:")
        self.title.pack(padx=10, pady=(25, 10))
        
        self.link = ctk.CTkEntry(self.root, width=350, height=40, textvariable=self.user_input)
        self.link.pack()
        self.link.bind("<Return>", self.on_enter_key)

        self.finishLabel = ctk.CTkLabel(self.root, text="No download started")
        self.finishLabel.pack(padx=10, pady=10)

        self.download = ctk.CTkButton(self.root, text="Grab next image", command=self.startDownload)
        self.download.pack(padx=10, pady=10)

        self.skip = ctk.CTkButton(self.root, text="Skip image segment", command=self.skipSegment)
        self.skip.pack(padx=10, pady=10)
        self.skip.pack_forget()

        self.image_display = ctk.CTkToplevel()
        self.image_display.geometry('720x240')
        self.image_display.title('Transcribe this image')
        self.image_display.line = ctk.CTkLabel(self.image_display, text="")    
        self.image_display.line.pack()
        self.image_display.withdraw()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def skipSegment(self):
        self.curr_segment += 1
        
        if os.path.exists(self.build_segment_path()):
            self.openImage(self.build_segment_path(), self.image_display.line)
        else:            
            self.checkForMore()

    def on_close(self):
        trash = ['temp.jpg', 'log.csv']
        for path in trash:
            if os.path.exists(path):
                os.unlink(path)
        if os.path.exists('segmented'):
            shutil.rmtree('segmented')        
        
        if hasattr(self, 'image_display') and self.image_display:
            self.image_display.destroy()

        for after_id in root.tk.eval('after info').split():
            root.after_cancel(after_id)
        
        self.root.destroy()

    def parse_training_log(self):
        url = 'https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/log.csv'
        response = requests.get(url)
        with open('log.csv', 'wb') as f:
            f.write(response.content)
        vols = []
        with open('log.csv', 'r') as f:
            for line in f:            
                data = line.split(',')
                try:
                    vols.append({'id': int(data[0]), 'last_im': int(data[1])})
                except:
                    continue
        return vols

    def update_training_log(self, volume, image):
        vols = self.parse_training_log()
        found_vol = False
        for index, vol in enumerate(vols):
            if str(vol['id']) == volume:
                vols[index]['last_im'] = image
                found_vol = True
        if not found_vol:
            vols.append({'id': volume, 'last_im': image})
        data = ''
        for vol in vols:
            data += str(vol['id']) + ',' + str(vol['last_im']) + '\n'
        headers = {"Content-Type":"text/csv"}
        requests.put('https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/log.csv', data=data, headers=headers)

    def startDownload(self):        
        self.vol_id = self.user_input.get()        
        
        for vol in self.parse_training_log():
            if str(vol['id']) == self.vol_id:
                self.img_id = vol['last_im'] + 1
                break        
            
        if self.img_id == None:
            self.img_id = 1

        str_im = '0' * (4 - len(str(self.img_id))) + str(self.img_id)
        obj_key = self.vol_id + '-' + str_im + '.jpg'
        url = 'https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-production-jpgs/' + obj_key
        response = requests.get(url)

        if response.status_code == 200:
            with open('temp.jpg', 'wb') as f:
                f.write(response.content)
            # clumsy fix, check why always 200 later
            size = os.path.getsize('temp.jpg')
            if size > 10000:
                self.finishLabel.configure(text="Image " + str(self.img_id) + " downloaded!", text_color='green')
                self.link.pack_forget()
                self.download.configure(text="Begin transcription", command=self.beginTranscription)
                return        
    
        self.finishLabel.configure(text="Download error", text_color='red')

    def downloadNext(self):        
        self.img_id += 1

        str_im = '0' * (4 - len(str(self.img_id))) + str(self.img_id)
        obj_key = self.vol_id + '-' + str_im + '.jpg'
        url = 'https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-production-jpgs/' + obj_key
        response = requests.get(url)

        if response.status_code == 200:
            with open('temp.jpg', 'wb') as f:
                f.write(response.content)
            # clumsy fix, check why always 200 later
            size = os.path.getsize('temp.jpg')
        if (size < 10000) or (response.status_code != 200):
            return False
        else:            
            self.finishLabel.configure(text="Image " + str(self.img_id) + " downloaded!", text_color='green')
            self.root.update()
        
        return True

    def beginTranscription(self):
        self.skip.pack()
        self.finishLabel.configure(text="Processing image", text_color='green')
        self.root.update()
        self.update_training_log(self.vol_id, self.img_id)
        self.link.pack(before=self.finishLabel)
        self.link.delete(0, tk.END)
        
        check = driver(self.vol_id, self.img_id)
        if not check: 
            self.checkForMore()
                
        self.curr_segment = 1    
        
        # implement zoomable image
        
        self.image_display.deiconify()    
        self.openImage(self.build_segment_path(), self.image_display.line)    
        self.title.configure(text="Transcribe text here:")
        self.finishLabel.configure(text="")
        self.download.configure(text="Save transcription", command=self.saveTranscription)

    def build_segment_path(self):
        next_segment = self.vol_id + '-' + ('0' * (4 - len(str(self.img_id))) + str(self.img_id)) + '-' + ('0' * (2 - len(str(self.curr_segment))) + str(self.curr_segment))
        return os.path.join('segmented', next_segment + '.jpg')

    def saveTranscription(self):
        if len(self.user_input.get()) < 1:
            self.finishLabel.configure(text="No transcription entered.", text_color='red')
            return
        
        with open(self.build_segment_path(), "rb") as f:
            img_data = f.read()
        headers = {"Content-Type":"image/jpeg"}
        requests.put("https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/" + self.build_segment_path()[self.build_segment_path().find('\\') +1:], data=img_data, headers=headers)
        requests.put("https://zoqdygikb2.execute-api.us-east-1.amazonaws.com/v1/ssda-htr-training/" + self.build_segment_path()[self.build_segment_path().find('\\') +1:self.build_segment_path().find('.')] + '.txt', data=self.user_input.get())
        self.link.delete(0, tk.END)
        
        self.curr_segment += 1
        
        if os.path.exists(self.build_segment_path()):
            self.openImage(self.build_segment_path(), self.image_display.line)
        else:            
            self.checkForMore()

    def checkForMore(self):
        more_images = self.downloadNext()
        if not more_images:
            self.image_display.withdraw()
            self.title.configure(text="Volume completed! Enter another SSDA volume ID to transcribe from:")
            self.finishLabel.configure(text="No download started", text_color='black')
            self.download.configure(text="Grab next image", command=self.startDownload)
            self.skip.pack_forget()
        else:
            self.beginTranscription()

    def openImage(self, image_path, display_widget):
        img = Image.open(image_path)
        im = ctk.CTkImage(img, size=(img.width, img.height))        
        display_widget.configure(image=im)
        img.close()

    def on_enter_key(self, event):
        if self.download._text == "Grab next image":
            self.startDownload()
        elif self.download._text == "Begin transcription":
            self.beginTranscription()
        else:
            self.saveTranscription()

root = ctk.CTk()
app = TrGUI(root)
root.mainloop()