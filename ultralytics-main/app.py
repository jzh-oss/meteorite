import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F 

from .ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO("./runs/classify/train14/weights/best.pt")
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]


st.title("陨石检测器")
st.write("请上传一张图像，系统将判断其中是否包含陨石。")

uploaded_file = st.file_uploader("选择图像", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图像", use_column_width=True)


    model = load_model()


    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        print(len(output))
        probs = output[0].probs.data
        print(probs)
        confidence, predicted = torch.max(probs, 0)

    # 显示结果
    class_names = ["非陨石", "陨石"]
    st.write(f"预测结果：**{class_names[predicted.item()]}**")
    st.write(f"置信度：{confidence.item():.2f}")
