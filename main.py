from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import numpy as np
from io import BytesIO
import tempfile

app = FastAPI()

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # Detect edges
    edges = cv2.Canny(blur, 100, 200)
    
    # Save processed image to temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        cv2.imwrite(temp_file.name, edges)
        
    return FileResponse(temp_file.name, media_type="image/jpeg", filename="processed_image.jpg")


    !uvicorn main:app --reload --host 0.0.0.0 --port 8000
