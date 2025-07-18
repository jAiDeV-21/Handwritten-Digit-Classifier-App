import streamlit as st
import numpy as np
from PIL import Image
import pickle

# ----- Load model -----
@st.cache_resource
def load_model(path="mnist_model.pkl"):
    with open(path, "rb") as f:
        sizes, weights, biases = pickle.load(f)
    net = Network(sizes)
    net.weights = weights
    net.biases = biases
    return net

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = []
        self.biases = []

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# ----- Image preprocessing -----
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize((28, 28))
    img_array = np.array(image).astype(np.float32)
    
    # Invert if needed
    if img_array.mean() > 127:
        img_array = 255 - img_array
        
    img_array /= 255.0
    return img_array.reshape(784, 1)


st.set_page_config(page_title="Digit Classifier (From Scratch)", layout="centered")
st.title("🧠 Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload a 28x28 pixel image of a digit (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    if st.button("Predict"):
        net = load_model()
        input_vector = preprocess_image(image)
        output = net.feedforward(input_vector)
        predicted_digit = int(np.argmax(output))

        st.success(f"🧮 Predicted Digit: **{predicted_digit}**")
        st.bar_chart(output.reshape(-1))

