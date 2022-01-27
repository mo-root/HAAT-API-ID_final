from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import io
import requests
import pytesseract
from PIL import Image, ImageEnhance

global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

UPLOAD_FOLDER = r'C:\Users\Student\Desktop\HAAT-id-api\static\img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, template_folder='templates', static_folder='static')  # Bootstrap(app)

app.config['SECRET_KEY'] = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def split(word):
    arr = [char for char in word]
    print(arr)
    i = 0
    arr_clean = list()
    for elm in arr:
        try:
            float(elm)
            print("could convert string to float:", elm)
            arr_clean.append(elm)
        except ValueError as e:
            print(e)
    print(arr_clean)

    # using list comprehension
    IDD = ''.join(map(str, arr_clean))

    return IDD


camera = cv2.VideoCapture(0)


def test(p):
    img = Image.open(p)
    width, height = img.size

    img.show()

    enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.5)
    text = pytesseract.image_to_string(img)
    print('THis s the text with the link', text)
    fID = split(text)
    Actual_ID = "214549297"
    if Actual_ID in fID:
        return fID
        print(fID, " has the ID")
    if Actual_ID not in fID:
        img = img.transpose(Image.ROTATE_90)
        img.show()
        text = pytesseract.image_to_string(img)
        # ('HOLD UP')
        # print(pytesseract.image_to_boxes(img))

        print('This s the text with the link but flipped', text)
        fID = split(text)
        if (Actual_ID in fID):
            return fID
        if Actual_ID not in fID:
            img = img.transpose(Image.ROTATE_90)
            img.show()
            text = pytesseract.image_to_string(img)
            ('HOLD UP')
            print(pytesseract.image_to_boxes(img))

            print('This s the text with the link but flipped', text)
            fID = split(text)
            if (Actual_ID in fID):
                return fID

        if Actual_ID not in fID:
            img = img.transpose(Image.ROTATE_90)
            img.show()
            text = pytesseract.image_to_string(img)
            ('HOLD UP')
            print(pytesseract.image_to_boxes(img))

            print('This s the text with the link but flipped', text)
            fID = split(text)
            if (Actual_ID in fID):
                return fID
        elif fID == '':
            return "very bad image"


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read()
        frame = cv2.flip(frame,1)
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                print("heereee is the f path", p)
                cv2.imwrite(p, cv2.flip(frame,1))
                test(p)

            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/idl', methods=['GET', 'POST'])
def checking():
    if request.method == 'POST':
        imgL = request.form['image']
        response = requests.get(imgL)
        img = Image.open(io.BytesIO(response.content))
        width, height = img.size

        # left = width / 1.5
        # top = height / 1.3
        # right = width
        # bottom = height
        # img = img.crop((left, top, right, bottom))

        img.show()

        # color_thief = ColorThief(io.BytesIO(response.content))
        #

        # get the dominant color

        # dominant_color = color_thief.get_palette(6)
        # print("color goes here", dominant_color)

        enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.5)
        text = pytesseract.image_to_string(img)
        print('THis s the text with the link', text)
        fID = split(text)
        Actual_ID = "214549297"
        if (Actual_ID in fID):
            return fID
            print(fID, " has the ID")
        if Actual_ID not in fID:
            img = img.transpose(Image.ROTATE_90)
            img.show()
            text = pytesseract.image_to_string(img)
            # ('HOLD UP')
            # print(pytesseract.image_to_boxes(img))

            print('This s the text with the link but flipped', text)
            fID = split(text)
            if (Actual_ID in fID):
                return fID
            else:
                img = img.transpose(Image.ROTATE_90)
                img.show()
                text = pytesseract.image_to_string(img)
                ('HOLD UP')
                print(pytesseract.image_to_boxes(img))

                print('This s the text with the link but flipped', text)
                fID = split(text)
                if (Actual_ID in fID):
                    return fID

            if Actual_ID not in fID:
                img = img.transpose(Image.ROTATE_90)
                img.show()
                text = pytesseract.image_to_string(img)
                ('HOLD UP')
                print(pytesseract.image_to_boxes(img))

                print('This s the text with the link but flipped', text)
                fID = split(text)
                if (Actual_ID in fID):
                    return fID
            else:
                return "Please enter another Image"
        return fID
    return render_template('About.html')


@app.route('/idc', methods=['GET', 'POST'])
def capturing_cam():
    if request.method == 'POST':
        photo = request.files['image']

        upload_file(photo)
        print("asddddddddd")
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
        width, height = img.size

        # left = width / 1.5
        # top = height / 1.3
        # right = width
        # bottom = height
        # img = img.crop((left, top, right, bottom))

        img.show()

        # color_thief = ColorThief(io.BytesIO(response.content))
        #

        # get the dominant color

        # dominant_color = color_thief.get_palette(6)
        # print("color goes here", dominant_color)

        enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.5)
        text = pytesseract.image_to_string(img)
        print('THis s the text with the link', text)
        fID = split(text)
        Actual_ID = "214549297"
        if Actual_ID in fID:
            return fID
            print(fID, " has the ID")
        if Actual_ID not in fID:
            img = img.transpose(Image.ROTATE_90)
            img.show()
            text = pytesseract.image_to_string(img)
            # ('HOLD UP')
            # print(pytesseract.image_to_boxes(img))

            print('This s the text with the link but flipped', text)
            fID = split(text)
            if (Actual_ID in fID):
                return fID
            if Actual_ID not in fID:
                img = img.transpose(Image.ROTATE_90)
                img.show()
                text = pytesseract.image_to_string(img)
                ('HOLD UP')
                print(pytesseract.image_to_boxes(img))

                print('This s the text with the link but flipped', text)
                fID = split(text)
                if (Actual_ID in fID):
                    return fID

            if Actual_ID not in fID:
                img = img.transpose(Image.ROTATE_90)
                img.show()
                text = pytesseract.image_to_string(img)
                ('HOLD UP')
                print(pytesseract.image_to_boxes(img))

                print('This s the text with the link but flipped', text)
                fID = split(text)
                if (Actual_ID in fID):
                    return fID
            elif fID == '':
                return "very bad image"
        return fID, 'Please enter another Image'
    return render_template('Pics.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def upload_file(file):
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     