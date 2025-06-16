from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
model = YOLO("/root/jzh/demo_2/ultralytics-main/ultralytics/cfg/models/11/yolo11-cls-resnet50.yaml")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
# 
# Train the model
results = model.train(data="/root/jzh/meteorite/meteorite_process",augment=True ,mixup= 0.3,epochs=10000)

# from ultralytics import YOLO

# if __name__ == '__main__':
#     # 加载预训练的 YOLO 模型
#     model = YOLO('/root/jzh/other/yolo11x-pose.pt')  # 这里是预训练的 YOLOv12 模型路径
#     print("✅ Model loaded, starting training...")
#     total_params = sum(p.numel() for p in model.model.parameters()) 
#     model.train(data='/root/jzh/demo_2/ultralytics-main/pos.yaml',
#                 cache=False,
#                 epochs=1000,
#                 batch=32,
#                 close_mosaic=0,
#                 device='0',
#                 optimizer='SGD',
#                 project='runs/train_1w-pose',
#                 name='exp',
#                 )