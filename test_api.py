import requests
import os

def test_api_endpoint(api_url, video_path, prompt):
    """
    Test the FastAPI endpoint exactly like the hackathon evaluation
    """
    try:
        # Prepare the files and data
        with open(video_path, 'rb') as video_file:
            files = {
                'video': ('test_video.mp4', video_file, 'video/mp4')
            }
            data = {
                'prompt': prompt
            }
            
            print(f"🚀 Testing API endpoint: {api_url}")
            print(f"📹 Video file: {video_path}")
            print(f"❓ Prompt: {prompt}")
            print("⏳ Sending request...")
            
            # Make the request
            response = requests.post(
                f"{api_url}/infer",
                files=files,
                data=data,
                timeout=60
            )
            
            print(f"✅ Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print("🎉 SUCCESS!")
                print(f"📝 Response: {response.text[:500]}...")
                return True
            else:
                print("❌ FAILED!")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_health_check(api_url):
    """Test health check endpoint"""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(response.json())
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"  # Change this to your deployed URL
    VIDEO_PATH = "test_video.mp4"      # Path to your test video
    TEST_PROMPT = "What action is happening in this clip?"
    
    print("🧪 FastAPI Hackathon Submission Tester")
    print("=" * 50)
    
    # Test health check first
    print("\n1. Testing Health Check...")
    health_ok = test_health_check(API_URL)
    
    if health_ok:
        print("\n2. Testing Main Endpoint...")
        
        # Check if video file exists
        if os.path.exists(VIDEO_PATH):
            success = test_api_endpoint(API_URL, VIDEO_PATH, TEST_PROMPT)
            
            if success:
                print("\n🎉 ALL TESTS PASSED!")
                print("Your API is ready for hackathon submission!")
            else:
                print("\n❌ API test failed")
        else:
            print(f"❌ Test video not found: {VIDEO_PATH}")
            print("Please add a test video file or update VIDEO_PATH")
    else:
        print("❌ Health check failed - API might not be running")
    
    print("\n" + "=" * 50)
    print("Ready to submit? Use this cURL command:")
    print(f"""
curl -X POST "{API_URL}/infer" \\
  -H "accept: text/plain" \\
  -F "video=@{VIDEO_PATH}" \\
  -F "prompt=What action is happening in this clip?"
""")