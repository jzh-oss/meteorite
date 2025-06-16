from ultralytics import YOLO
from PIL import Image

# Load a model
model = YOLO("/root/jzh/demo_2/ultralytics-main/runs/classify/train13/weights/best.pt")  # load a custom model

# Predict with the model
results = model("/root/jzh/meteorite/meteorite_process/val/No/000001.jpg",visualize=False)  # predict on an image
# print(results,"len(results)")
# Access the results
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)

for r in results:
    print(r.probs)
 
    im_array = r.plot() # plot a BGR numpy array of predictions
 
    im = Image.fromarray(im_array[..., ::-1]) # RGB PIL image
 
    im.show() # show image
 
    im.save('/root/jzh/data/predict_image/result06.jpg') # save image