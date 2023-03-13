from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import onnxruntime as ort

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*',
                   # 'http://localhost.tiangolo.com',
                   # # 'https://localhost.tiangolo.com',
                   # 'http://localhost',
                   # 'http://localhost:8080',
                   ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

IMAGE = cv2.imread('image.jpg')
SESSION = ort.InferenceSession(
    'models/3d_shape_detection_model.onnx', providers=['CPUExecutionProvider'])
NAMES = ['cylinder', 'cone', 'sphere']


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def detect(im):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in SESSION.get_outputs()]
    outname

    inname = [i.name for i in SESSION.get_inputs()]
    inname

    inp = {inname[0]: im}

    # ONNX inference
    outputs = SESSION.run(outname, inp)[0]

    predictions = []

    for _, x0, y0, x1, y1, cls_id, score in outputs:
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()

        cls_id = int(cls_id)
        name = NAMES[cls_id]

        score = round(float(score), 3)

        predictions.append({
            'box': box,
            'name': name,
            'score': score,
        })

    return predictions


@app.get('/')
def read_root():
    return {'message': 'Hello, world!'}


@app.get('/detect-test/')
def read_test():
    predictions = detect(IMAGE)
    return predictions


@app.post('/detect/')
async def create_detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    predictions = detect(im)
    return predictions