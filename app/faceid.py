"""Builds the app"""

# Import kivy dependencies 
from kivy.app import App
from kivy.core.window import Window  
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import os
import PIL
# from PIL import Image
import cv2
import torch
from torch import nn
import numpy as np
from torchvision import transforms
from layers import SiameseModel

# Set window properties
Window.clearcolor = (0.95, 0.95, 0.95, 1) 
# Window.size = (400, 600)                
Window.title = "Face Verification App"    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"E:\ROOT\FCAI\ML&DL_Courses\Projects\Deep_Facial_Recognition\app\models\Siamese_Model.pth"
VERIFICATION_IMAGES_PATH = r"E:\ROOT\FCAI\ML&DL_Courses\Projects\Deep_Facial_Recognition\app\application_data\verification_images"
INPUT_IMAGE_PATH = r"E:\ROOT\FCAI\ML&DL_Courses\Projects\Deep_Facial_Recognition\app\application_data\input_image"


# Build App and layout
class MainApp(App):
    """Main application class for the Kivy app"""

    def __init__(self):
        """Initialize the app"""
        super(MainApp, self).__init__()
        # main layout
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Verify", 
                             on_press=self.verify, 
                             size_hint=(1,.1),
                             background_color=(0.2, 0.6, 1, 1),  
                             color=(1, 1, 1, 1),                 
                             font_size=20,                       
                             bold=True                                                     
                             )
        
        self.verification_label = Label(text="Verification Uninitiated", 
                                        size_hint=(1,.1),
                                        color=(0, 0, 0, 1),     
                                        font_size=18,           
                                        bold=True               
                                      )
        self.capture = cv2.VideoCapture(0)

        self.model = SiameseModel()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        


    def build(self):
        """Build the main layout of the app"""
       
        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        """Update the webcam feed and process the image"""

        # Read frame from opencv
        _, frame = self.capture.read()
        frame = frame[0:250, 200:450, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def process(self, image_path):
        """Process the input image and return the transformed tensor"""
        data_transform = transforms.Compose([
        transforms.Resize(size=(105, 105)), # resize to 105x105
        transforms.ToTensor() # convert to tensor and normalize to [0,1]
        ])
        
        img = PIL.Image.open(image_path).convert("RGB")
        img = data_transform(img) 
        return img.to(device)  # [3, 105, 105]


    def make_predictions(self, model: nn.Module,
                     data: list,
                     device: torch.device = device):
        pred_probs = []
        model.to(device)
        model.eval()
        with torch.inference_mode():
            for sample in data:
                # Prepare sample
                in_sample = torch.unsqueeze(sample[0], dim=0).to(device)
                val_sample = torch.unsqueeze(sample[1], dim=0).to(device)

                # Forward pass (model outputs)
                pred_logits = model(in_sample, val_sample)

                # Get prediction probabilities (logits -> probabilities)
                pred_prob = torch.sigmoid(pred_logits.squeeze())

                # Get pred_prob off the GPU
                pred_probs.append(pred_prob.cpu())
        
        # Stack the pre_probs to turn  list into a tensor
        return torch.stack(pred_probs)


    # Verification function
    def verify(self, model):
        detection_threshold=0.9
        verification_threshold=0.8


        SAVE_PATH = os.path.join(INPUT_IMAGE_PATH, 'input_image.jpg') 
        _, frame = self.capture.read()
        frame = frame[0:250, 200:450, :]
        cv2.imwrite(SAVE_PATH, frame)


        # Build results array
        results = []
        self.model.eval()
        for image_path in os.listdir(VERIFICATION_IMAGES_PATH):

            # Read the verification image and input image and process them (resize, normalize and convert to tensor)
            verification_image_path = os.path.join(VERIFICATION_IMAGES_PATH, image_path)  
            input_image_path   = os.path.join(INPUT_IMAGE_PATH, 'input_image.jpg')            
            verification_image = self.process(verification_image_path)
            input_image = self.process(input_image_path)
            # Make predictions
            data=[(verification_image, input_image)] # [ (verification_image, input_image) ] 
            pred_probs = self.make_predictions(model=self.model,
                                        data=data,
                                        device=device)
            results.append(pred_probs.item())
        
        # calculate the detection results and verification results
        detection_results = torch.sum(torch.tensor(results) > detection_threshold)
        verification_results = detection_results / len(results)
        verified = verification_results > verification_threshold

        # Set verification text
        # self.verification_label.text = (
        #     "Mohammad " if verified else "Not Recognized "
        # )

        self.verification_label.text = (
            "[b][color=00AA00]Welcome Taha[/color][/b]" if verified 
            else "[b][color=FF0000]Not Recognized[/color][/b]"
        )
        self.verification_label.markup = True


        Logger.info(results)
        Logger.info(detection_results)
        Logger.info(verification_results)
        Logger.info(verified)

        return results, verified


if __name__ == '__main__':
    MainApp().run()