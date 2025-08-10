from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import google.generativeai as genai
from PIL import Image
import cv2
import tempfile
import os
import base64
import io
import time
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Visual Understanding Chat Assistant",
    description="AI-powered video analysis using Gemini Pro Vision",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MAX_FRAMES = 10  # Maximum frames to analyze
FRAME_SAMPLE_RATE = 90  # Extract frame every 90 frames (3 seconds at 30fps)

# Configure Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    model = None

def extract_video_frames(video_path: str, max_frames: int = MAX_FRAMES) -> list:
    """
    Extract frames from video file
    Returns list of PIL Images
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            cap.release()
            return []
        
        # Calculate which frames to extract (evenly distributed)
        if frame_count <= max_frames:
            frame_indices = list(range(frame_count))
        else:
            step = frame_count / max_frames
            frame_indices = [int(i * step) for i in range(max_frames)]
        
        # Extract frames
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize if too large (for API efficiency)
                if pil_image.width > 1024 or pil_image.height > 1024:
                    pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                frames.append(pil_image)
        
        cap.release()
        return frames
    
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        raise Exception(f"Error extracting frames: {str(e)}")

def analyze_with_gemini(frames: list, prompt: str) -> str:
    """
    Analyze frames with Gemini Pro Vision
    """
    try:
        if not model:
            return "Error: Gemini API not configured. Please set GOOGLE_API_KEY environment variable."
        
        if not frames:
            return "Error: No valid frames could be extracted from the video."
        
        # Create comprehensive analysis prompt
        analysis_prompt = f"""
        You are an expert video analyst. I'm providing you with {len(frames)} key frames from a video.
        
        User Question/Prompt: "{prompt}"
        
        Please analyze these video frames and provide a detailed, accurate response. Focus on:
        1. What's happening in the video
        2. Key activities, people, objects, and scenes
        3. Any progression or changes between frames
        4. Specific details relevant to the user's question
        
        Provide a clear, concise response that directly answers the user's question.
        """
        
        # Prepare content for Gemini (prompt + images)
        content = [analysis_prompt] + frames
        
        # Generate response
        response = model.generate_content(content)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Error: Could not generate response from Gemini API."
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle common API errors
        if "api_key_invalid" in error_msg or "invalid api key" in error_msg:
            return "Error: Invalid Google AI API key."
        elif "quota_exceeded" in error_msg or "quota" in error_msg:
            return "Error: API quota exceeded. Please try again later."
        elif "safety" in error_msg or "blocked" in error_msg:
            return "Error: Content was blocked by safety filters."
        else:
            return f"Error: {str(e)}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Visual Understanding Chat Assistant API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "gemini_configured": model is not None,
        "timestamp": time.time()
    }

@app.post("/infer", response_class=PlainTextResponse)
async def infer(
    video: UploadFile = File(..., description="Video file to analyze"),
    prompt: str = Form(..., description="Question or prompt about the video")
):
    """
    Main inference endpoint for hackathon evaluation
    
    Required format:
    - POST request with multipart/form-data
    - video: uploaded video file
    - prompt: string question/prompt
    - Response: plain text
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video file provided")
        
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        # Check if API key is configured
        if not GOOGLE_API_KEY or not model:
            return "Error: Gemini API not configured. Please set GOOGLE_API_KEY environment variable."
        
        # Validate video file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_extension = os.path.splitext(video.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            return f"Error: Unsupported video format. Supported formats: {', '.join(allowed_extensions)}"
        
        # Create temporary file to save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            # Read and save video content
            content = await video.read()
            tmp_file.write(content)
            tmp_video_path = tmp_file.name
        
        try:
            # Extract frames from video
            frames = extract_video_frames(tmp_video_path, MAX_FRAMES)
            
            if not frames:
                return "Error: Could not extract any frames from the video. Please check the video file."
            
            # Analyze with Gemini
            response = analyze_with_gemini(frames, prompt)
            
            # Log performance
            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f}s - Frames: {len(frames)} - Prompt: {prompt[:50]}...")
            
            return response
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_video_path)
            except:
                pass
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Unexpected error after {processing_time:.2f}s: {str(e)}"
        print(f"ERROR: {error_msg}")
        return f"Error: {str(e)}"

@app.post("/test")
async def test_endpoint(
    video: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Test endpoint that returns JSON (for debugging)
    """
    start_time = time.time()
    
    try:
        # Save temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            tmp_video_path = tmp_file.name
        
        # Extract frames
        frames = extract_video_frames(tmp_video_path, MAX_FRAMES)
        
        # Analyze with Gemini
        response = analyze_with_gemini(frames, prompt)
        
        # Clean up
        os.unlink(tmp_video_path)
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "response": response,
            "frames_analyzed": len(frames),
            "processing_time": f"{processing_time:.2f}s",
            "prompt": prompt,
            "video_filename": video.filename
        }
    
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "processing_time": f"{processing_time:.2f}s"
        }

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )