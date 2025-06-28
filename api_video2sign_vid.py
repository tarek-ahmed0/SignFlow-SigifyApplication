from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import uuid

# Our Models :
from text_preprocessing import get_original_text
from translation_pipeline import coll2formal_translation
from sentiment_analysis_pipeline import sentiment_analysis

app = FastAPI()

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    try:
        temp_filename = f"temp_{uuid.uuid4().hex}.mp4"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = get_original_text(temp_filename)
        if not isinstance(extracted_text, str) or extracted_text.startswith("Error"):
            os.remove(temp_filename)
            return {"error": "Failed to extract text from video", "details": extracted_text}

        transformed_text = coll2formal_translation(extracted_text)
        if not isinstance(transformed_text, str):
            os.remove(temp_filename)
            return {"error": "Translation failed"}

        try:
            words, signids = sentiment_analysis(transformed_text)
        except Exception as e:
            os.remove(temp_filename)
            return {"error": "Sentiment analysis failed", "details": str(e)}

        os.remove(temp_filename) 
        return {
            "Video Original Text": extracted_text,
            "Transformed Text": transformed_text,
            "The Nearest Words Of Stored": words,
        }

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content = {"error": "Unexpected server error", "details": str(e)},
        )
