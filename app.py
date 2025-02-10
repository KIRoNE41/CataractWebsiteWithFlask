from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os, shutil
import cv2
import numpy as np
import onnx
import torch
from PIL import Image
from io import BytesIO
from starlette.requests import Request

torch.set_grad_enabled(False)  # Disable gradients to save memory

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load ONNX models
face_model = onnx.load("model/FaceOD.onnx")
eye_model = onnx.load("model/EyeOD.onnx")
cataract_model = onnx.load("model/CataractOD.onnx")

# Create a session for each model
import onnxruntime
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
face_session = onnxruntime.InferenceSession(face_model,providers=providers)
eye_session = onnxruntime.InferenceSession(eye_model,providers=providers)
cataract_session = onnxruntime.InferenceSession(cataract_model,providers=providers)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
global diagnosis_dict
diagnosis_dict = {0: 'ไม่พบเจอดวงตา', 1: 'ไม่พบเจอดวงตา'}


def reDIR():
    shutil.rmtree(RESULT_FOLDER)
    shutil.rmtree(UPLOAD_FOLDER)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)


def preprocess_image(image_input):
    # Preprocessing image for ONNX model
    image = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize for model input
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    image = image.astype(np.float32)  # Ensure float32
    image /= 255.0  # Normalize the image to 0-1 range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def run_onnx_model(model_session, image):
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    result = model_session.run([output_name], {input_name: image})
    return result


def cataract_detect(image_input, num):
    global diagnosis_dict
    image = preprocess_image(image_input)  # Prepare image for ONNX model
    results = run_onnx_model(cataract_session, image)  # Run cataract model

    # Extract probabilities for each class
    prob_normal = results[0][0]  # Probability of "Normal"
    prob_cataract = results[0][1]  # Probability of "Cataract"

    # Determine the diagnosis based on higher probability
    if prob_cataract > prob_normal:
        diagnosis = "ดวงตาเป็นโรคต้อกระจก"
    else:
        diagnosis = "ดวงตาปกติ"

    # Save the diagnosis in a dictionary
    diagnosis_dict[num] = diagnosis  
    cv2.imwrite(f'{RESULT_FOLDER}/cataract_{num}.jpg', image_input)  # Save image
    
    return diagnosis


def face_detect(image_input):
    results = run_onnx_model(face_session, preprocess_image(image_input))
    # Assuming the model detects faces and provides bounding boxes

    # Process the detected faces and perform further analysis on eyes
    eye_detect(image_input)


def eye_detect(image_input):
    global eye_num
    results = run_onnx_model(eye_session, preprocess_image(image_input))
    # Assuming the model detects eyes and provides bounding boxes

    cataract_detect(image_input, eye_num)  # Get diagnosis for each eye


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    reDIR()
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    global diagnosis_dict
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    source = cv2.imread(filepath)
    face_detect(source)
    return templates.TemplateResponse("index.html", 
                                      {"request": request,
                                       "img": "results_eye.jpg", 
                                       "img2": "results0.jpg", 
                                       "img3": "results1.jpg",
                                       "diagnosis1": str(diagnosis_dict.get(0)), 
                                       "diagnosis2": str(diagnosis_dict.get(1))})


@app.post("/capture", response_class=HTMLResponse)
async def capture(request: Request, file: UploadFile = File(...)):
    global diagnosis_dict
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    source = cv2.imread(filepath)
    face_detect(source)
    return templates.TemplateResponse("index.html", 
                                      {"request": request, 
                                       "img": "results_eye.jpg", 
                                       "img2": "results0.jpg", 
                                       "img3": "results1.jpg",
                                       "diagnosis1": str(diagnosis_dict.get(0)), 
                                       "diagnosis2": str(diagnosis_dict.get(1))})


@app.get("/result", response_class=HTMLResponse)
async def view_image(request: Request):
    global diagnosis_dict
    return templates.TemplateResponse("index.html", 
                                      {"request": request, 
                                       "img": "results_eye.jpg", 
                                       "img2": "results0.jpg", 
                                       "img3": "results1.jpg",
                                       "diagnosis1": str(diagnosis_dict.get(0)), 
                                       "diagnosis2": str(diagnosis_dict.get(1))})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get PORT from environment, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
