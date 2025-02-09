from flask import Flask, render_template, request, send_file
import os, shutil
import cv2
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Initialize models
model1 = YOLO("model/FaceOD.pt")
model2 = YOLO('model/EyeOD.pt')
model3 = YOLO("model/CataractOD.pt")

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static','uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'static','results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
global diagnosis_dict
diagnosis_dict ={0:'ไม่พบเจอดวงตา',
                 1:'ไม่พบเจอดวงตา'}

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
            # Assuming "Cataract" corresponds to a particular class label in the model
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

# Modify the face_detect and eye_detect functions to handle diagnosis
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
            cv2.imwrite(RESULT_FOLDER+'/ultralytics_crop_face_' + str(i) + '.jpg', img)
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
            cataract_detect(img,i)  # Get diagnosis here
        r.save(filename=f"{RESULT_FOLDER}/results_eye.jpg")

# Route for index page and form to upload image
@app.route('/')
def index():
    reDIR()
    return render_template('Templates/index.html')

# Route to handle the image upload and processing
@app.route('/upload', methods=['POST'])
def upload():
    global diagnosis_dict
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Save the uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
    file.save(filepath)

    # Read the image
    source = cv2.imread(filepath)
    face_detect(source)
    print(str(diagnosis_dict.get(0)))
    return render_template('Templates/index.html', 
                           img="results_eye.jpg", 
                           img2="results0.jpg", 
                           img3="results1.jpg",
                           diagnosis1=str(diagnosis_dict.get(0)), 
                           diagnosis2=str(diagnosis_dict.get(1)))  # Modify as needed based on detection results

@app.route('/capture', methods=['POST'])
def capture():
    global diagnosis_dict
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Save the uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, "captured_image.jpg")
    file.save(filepath)

    # Read the image
    source = cv2.imread(filepath)
    face_detect(source)
    return render_template('Templates/index.html', 
                           img="results_eye.jpg", 
                           img2="results0.jpg", 
                           img3="results1.jpg",
                           diagnosis1=str(diagnosis_dict.get(0)), 
                           diagnosis2=str(diagnosis_dict.get(1)))

@app.route('/result')
def view_image():
    global diagnosis_dict
    # Return the result image and diagnosis
    return render_template('Templates/index.html', 
                           img="results_eye.jpg", 
                           img2="results0.jpg", 
                           img3="results1.jpg",
                           diagnosis1=str(diagnosis_dict.get(0)), 
                           diagnosis2=str(diagnosis_dict.get(1)))  # Modify as needed based on detection results


if __name__ == '__main__':
    app.run(debug=True)
