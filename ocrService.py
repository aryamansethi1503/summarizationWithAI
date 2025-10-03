from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import pytesseract
import io
import os

app = FastAPI()

@app.post("/ocr/")
async def extract_text_from_image(file: UploadFile = File(...)):
   if not file.content_type.startswith("image/"):
       raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

   try:
       image_bytes = await file.read()
       image = Image.open(io.BytesIO(image_bytes))
       text = pytesseract.image_to_string(image)

       return {"filename": file.filename, "text": text}
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"Error performing OCR: {str(e)}")


