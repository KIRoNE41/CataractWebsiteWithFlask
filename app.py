from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os, shutil
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
from io import BytesIO
from starlette.requests import Request

torch.set_grad_enabled(False)  # ปิดการใช้ Gradient เพื่อประหยัด RAM

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize models
model1 = YOLO("model/FaceOD.pt")
model2 = YOLO('model/EyeOD.pt')
model3 = YOLO("model/CataractOD.pt")

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


def cataract_detect(image_input, num):
    global diagnosis_dict
    results = model3.predict(image_input, conf=0.5)
    names = model3.names
    boxes = results[0].boxes.xyxy.tolist()
    diagnosis = 'ไม่สามารถตรวจจับได้'  # Default diagnosis

    for r in results:
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1]) 

        for c in r.boxes.cls:
            if names[int(c)] == "Cataract":
                diagnosis = 'ดวงตาเป็นโรคต้อกระจก'  # Update diagnosis if Cataract is detected
            if names[int(c)] == "Normal":
                diagnosis = 'ดวงตาปกติ'  # Update diagnosis if Cataract is detected
              

        # Save and process the image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            img = image_input[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(f'{RESULT_FOLDER}/ultralytics_crop_scan_{num}.jpg', img)
        diagnosis_dict[num] = diagnosis
        r.save(filename=f"{RESULT_FOLDER}/results{num}.jpg")

    return diagnosis  # Return the diagnosis

def face_detect(image_input):
    results = model1(image_input)
    names = model1.names
    boxes = results[0].boxes.xyxy.tolist()

    for r in results:
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        img = image_input

        for c3 in r.boxes.cls:
            print(names[int(c3)])

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            img = image_input[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(RESULT_FOLDER + '/ultralytics_crop_face_' + str(i) + '.jpg', img)
        r.save(filename=f"{RESULT_FOLDER}/results_face.jpg")
    eye_detect(img)

def eye_detect(image_input):
    global eye_num
    results = model2(image_input)
    names = model2.names
    boxes = results[0].boxes.xyxy.tolist()

    for r in results:
        im_bgr = r.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])

        for c in r.boxes.cls:
            print(names[int(c)])

        for i, box in enumerate(boxes):
            eye_num = i
            x1, y1, x2, y2 = box
            img = image_input[int(y1):int(y2), int(x1):int(x2)]
            img = cv2.resize(img, (int(img.shape[1] * 10), int(img.shape[0] * 10)))
            cv2.imwrite(RESULT_FOLDER + '/ultralytics_crop_eye_' + str(i) + '.jpg', img)
            cataract_detect(img, i)  # Get diagnosis here
        r.save(filename=f"{RESULT_FOLDER}/results_eye.jpg")

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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
