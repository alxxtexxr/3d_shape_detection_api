import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
import onnxruntime as ort

from config import MODEL_PATH
from utils import detect

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    '*',
    # 'http://localhost:3000',
  ],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)

session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

@app.get('/')
def read_root():
  return {'message': 'Hello, world!'}

@app.get('/detect/test')
def read_test():
  im = cv2.imread('image.jpg')
  predictions = detect(session, im)
  return predictions

@app.post('/detect/')
async def create_detect(file: UploadFile = File(...)):
  contents = await file.read()
  nparr = np.fromstring(contents, np.uint8)
  im = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  predictions = detect(session, im)
  return predictions

if __name__ == '__main__':
  uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)