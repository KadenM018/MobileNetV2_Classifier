import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
import os
from MobileNetV2 import MobileNetV2

# set variables
print("torch is available? : {}\n".format(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_name = "best_weights.pth"
weights_dir = "saves/expAll/weights"

# Set labels
classes = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
           10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J',
           20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T',
           30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z'}

# Transforms
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(224, antialias=True),
        torchvision.transforms.CenterCrop(224)
    ])

# Load model with weights
model = MobileNetV2(3, 36)
ckpt = torch.load(os.path.join(weights_dir, weights_name))
model.load_state_dict(ckpt['model'])
model = model.to(device)

# Set up capture
capture = cv2.VideoCapture(0)
capture.set(4,720)
 # Loop
working = True
frame_count = 0
pred = 0
score = 0

with torch.no_grad():
    model.eval()
    while working:
        # capture some images, convert them to PIL, then to model
        has_image, frame = capture.read()

        # put frames into model
        if frame_count == 4:
            # choose middle part to be detection area
            image = frame[100:450, 150:570]
            image_pil = Image.fromarray(image)
            img_tensor = transform(image_pil)
            img_tensor = img_tensor.float()
            img_tensor = img_tensor.to(device)
            img_tensor = img_tensor[None,:]
            all_pred = model(img_tensor)
            # see which result appeared the most
            all_pred = all_pred.detach().cpu().numpy()
            score = np.amax(all_pred)
            max_pred = np.argmax(all_pred, axis=1)
            pred = max_pred[0]
            frame_count = 0

        frame_count += 1

        # draw some data on frame
        cv2.putText(frame, "best guess: {}".format(classes[pred]), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
        cv2.putText(frame, "score: {:.3f}".format(score), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("Detection Result", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            working = False

 # End Loop
capture.release()
cv2.destroyWindow("Detection Result")
# Clean Up
