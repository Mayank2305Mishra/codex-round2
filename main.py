import os
import time
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
#from prompt import analysis_prompt
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from google import genai

# --- Configuration and Initialization ---

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Video Analysis API",
    description="An API to analyze video content based on a text prompt.",
    version="1.0.0"
)

# --- Generative AI Client Initialization ---

# This prompt is a simplified version of what might be in the original `prompt.py`
# It instructs the model on how to behave.

analysis_prompt = """
Analyze the video and answer the user's question. Be extremely concise and direct. Provide only the answer.
"""


try:
    # Initialize the Google Generative AI client
    #model = genai.GenerativeModel(model_name='gemini-2.5-flash')
    logger.info("Google Generative AI client initialized successfully.")
except Exception as e:
    logger.error(f"🚨 Failed to initialize Google API client: {e}")
    # We don't raise an exception here to allow the app to start,
    # but endpoints will fail if the client is not configured.
    model = None


# --- API Endpoint ---

@app.post("/infer", response_class=JSONResponse)
async def infer(
    video: UploadFile = File(..., description="The video file to be analyzed."),
    prompt: str = Form(..., description="The text prompt or question about the video.")
):
    """
    This endpoint receives a video and a text prompt, analyzes the video
    using a generative AI model, and returns a JSON response containing
    the prediction and the model's latency.
    """
    if not client:
        raise HTTPException(
            status_code=500,
            detail="Google API client is not configured. Check server logs for details."
        )

    # The genai library works with file paths, so we save the uploaded file to a temporary location.
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tfile:
            # Write the uploaded video content to the temporary file
            content = await video.read()
            tfile.write(content)
            temp_video_path = tfile.name
            logger.info(f"Video saved temporarily to: {temp_video_path}")

    except Exception as e:
        logger.error(f"Failed to save temporary video file: {e}")
        raise HTTPException(status_code=500, detail="Could not process uploaded video file.")

    try:
        # --- Video Processing with Google AI ---
        logger.info("Uploading video file to Google AI...")
        video_file = client.files.upload(file=temp_video_path)

        # Poll the file status until it is 'ACTIVE', meaning it's ready for use.
        while video_file.state.name == "PROCESSING":
            logger.info("Waiting for video to be processed...")
            time.sleep(5)  # Wait for 5 seconds before checking again
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            logger.error("Video processing failed on Google AI.")
            raise HTTPException(status_code=500, detail="Video processing failed.")

        logger.info("Video processed successfully. Generating content...")

        # --- Content Generation & Latency Calculation ---
        start_time = time.time()
        
        # Combine the user's prompt with the uploaded video for the model
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[f"{analysis_prompt}\n\n{prompt}", video_file])

        end_time = time.time()
        latency = end_time - start_time
        logger.info(f"Model generation latency: {latency:.4f} seconds")

        # Return the model's text response and latency in a JSON object
        logger.info("Successfully generated response from model.")
        
        response_data = {
            "prediction": response.text,
            "latency_seconds": round(latency, 4)
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"An error occurred during AI processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during AI processing: {e}")

    finally:
        # --- Cleanup ---
        # Ensure the temporary file is deleted after processing.
        if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"Cleaned up temporary file: {temp_video_path}")


# --- Main Execution Block ---

if __name__ == "__main__":
    # This allows running the app directly with `python app.py`
    # For production, it's recommended to use a WSGI server like Gunicorn:
    # gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
    uvicorn.run(app, host="0.0.0.0", port=8000)
