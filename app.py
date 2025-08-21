import os
import time
import traceback
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import torch
from faster_whisper import WhisperModel
import json
import re 

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all origins during development

# --- Global Initialization of Models ---
# This ensures models are loaded once when the application starts

# Determine the best device for Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Whisper will attempt to use device: {device} with compute type: {compute_type}")

# Whisper Model Initialization
whisper_model_instance = None # Initialize to None
try:
    whisper_model_instance = WhisperModel("tiny", device=device, compute_type=compute_type) #You can change "medium" to "large-v3" or any other model you have downloaded.
except Exception as e:
    print(f"Error loading Whisper model: {str(e)}")
    print("Whisper transcription will not be available.")
    print(traceback.format_exc()) # Print full traceback for model loading error

# Gemini API and Model Initialization
google_model = None # Initialize to None
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        google_model = genai.GenerativeModel('gemini-2.0-flash') # Use 'gemini-2.0-flash' for faster inference
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API or loading Gemini model: {e}")
        print("Gemini summarization will not be available.")
        print(traceback.format_exc()) # Print full traceback for Gemini setup error
else:
    print("GEMINI_API_KEY not found. Please set it as an environment variable.")
    print("Gemini summarization will not be available.")

# --- API Endpoints ---

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the Whisper model was loaded successfully
    if whisper_model_instance is None:
        return jsonify({"error": "Whisper model failed to load at startup. Cannot transcribe."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    original_filename = audio_file.filename
    print(f"Original audio filename received: {original_filename}")

    temp_audio_path = None # Initialize to None for finally block
    try:
        temp_filename = f"temp_audio_{os.urandom(8).hex()}_{original_filename}"
        temp_audio_dir = "temp_uploads"
        os.makedirs(temp_audio_dir, exist_ok=True) # Ensure directory exists
        temp_audio_path = os.path.join(temp_audio_dir, temp_filename)

        print(f"Saving temporary audio file to: {temp_audio_path}")
        audio_file.save(temp_audio_path)

        segments, info = whisper_model_instance.transcribe(temp_audio_path, beam_size=1) # Add the language parameter if needed, e.g., language="en"
        
        # segments = ["Dummy Text"]
        
        # class MockInfo:
        #     def __init__(self):
        #         self.duration = 270.0                                         # Use it when you use dummy text
        #         self.language = "en"
        # info = MockInfo()

        start_time = time.time() # Start time for calculation of transcription
        print("Transcription starts at:", start_time)
        
        transcribed_text_parts = []
        for segment in segments:
            transcribed_text_parts.append(segment.text) 
        
        # print(transcribed_text_parts)
        
        end_time = time.time()
        print("Transcription ends at:", end_time) 
        
        print(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        transcribed_text = " ".join(transcribed_text_parts)
        print(f"Joined text preview: {transcribed_text[:200]}...")
        print("Text joining complete.")

        # Format duration for readability
        minutes = int(info.duration // 60)
        seconds = int(info.duration % 60)
        formatted_duration = f"{minutes} minutes and {seconds} seconds"

        # Pass the original filename along with the duration and transcribed text
        # This will be the base text sent to the summarization endpoint
        text_with_metadata = (
            f"The following is a transcribed conversation from the audio file named '{original_filename}' "
            f"with an estimated duration of {formatted_duration}. Conversation: {transcribed_text}"
        )

        return jsonify({
            "transcribedText": text_with_metadata, 
        })

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        traceback.print_exc()
        error_message = str(e)
        if "ffmpeg" in error_message.lower() or "runtimeerror" in error_message.lower() and "failed to load" in error_message.lower():
            error_message = "Failed to process audio. This often indicates that FFmpeg is not installed or not in your system's PATH. Please install FFmpeg (https://ffmpeg.org/download.html) and ensure it's accessible."
        elif "tensor of 0 elements" in error_message.lower() or "could not decode audio" in error_message.lower():
            error_message = "Audio file appears to be empty or corrupted, or could not be decoded. Please try a different audio file. Ensure FFmpeg is installed."

        return jsonify({"error": f"Error during transcription: {error_message}"}), 500

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_audio_path}: {str(e)}")
                

@app.route('/summarize', methods=['POST'])
def generate_summary():
    if google_model is None:
        return jsonify({"error": "Gemini API not configured or model failed to load. Cannot summarize."}), 500

    data = request.get_json()
    transcribed_text_with_metadata = data.get('text') # This contains filename, duration, and conversation

    if not transcribed_text_with_metadata:
        return jsonify({"error": "No transcribed text provided"}), 400

    if len(transcribed_text_with_metadata.strip()) < 50:
        return jsonify({"summary": "The transcribed text is too short to generate a meaningful summary."}), 200

    # Consolidate all tasks into a single, comprehensive prompt
    combined_prompt = f"""
    You are an expert at analyzing transcribed audio conversations and providing structured insights.
    **YOUR RESPONSE MUST CONTAIN ONLY THE JSON OBJECT, NOTHING ELSE, NO INTRODUCTORY OR CONCLUDING REMARKS.**
    Please perform the following analyses on the provided conversation and present the results in a single, well-structured JSON object, enclosed in a markdown JSON code block (```json...```).

    --- Instructions ---
    1.  **General Summary**: Provide a concise summary focusing on key details, requests, and outcomes.
    2.  **Call Metadata**: Extract the original filename, estimated duration, and detected language. If information is not found, use "N/A".
    3.  **Sentiment Analysis**: Categorize the overall sentiment as "Positive", "Negative", "Neutral", or "Mixed", and provide a brief justification.
    4.  **Speaker Estimation**: Estimate the number of distinct speakers. If possible, infer speaking turns or approximate speaking time; otherwise, state that it's not definitively determinable from text alone.
    5.  **Cultural Context**: Analyze for any implied cultural context, norms, or specific references. State "No strong cultural context evident" if none is found.

    --- Expected JSON Output Format (within ```json...```) ---
    ```json
    {{
        "general_summary": "Your summary here.",
        "call_metadata": {{
            "filename": "audio_file.wav",
            "duration": "X minutes and Y seconds",
            "language": "en"
        }},
        "sentiment_analysis": {{
            "sentiment": "Positive/Negative/Neutral/Mixed",
            "justification": "Brief reason for sentiment."
        }},
        
        "speakers' duration": {{
            "speaker x ": "number of minutes/seconds speaker x spoke",
            "speaker y ": "number of minutes/seconds speaker y spoke."
        }},
        "speaker_estimation": "Estimated number of speakers and details.",
        "cultural_context": "Cultural insights or 'No strong cultural context evident'."
    }}
    ```

    --- Conversation ---
    {transcribed_text_with_metadata}
    --- End of Conversation ---
    """

    results = {}
    try:
        print("Generating all analyses using a single Gemini prompt...")
        response = google_model.generate_content(combined_prompt)
        
        # Add this print statement to see the raw response from Gemini
        print(f"Gemini raw response text: \n{response.text}")

        # Use regex to extract JSON content from markdown block
        json_match = re.search(r'```json\n(.*)\n```', response.text, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(1).strip()
            print(f"Extracted JSON string for parsing: \n{json_string}") # Debugging
            try:
                parsed_results = json.loads(json_string)
                results = parsed_results
                print("All analyses generated and parsed successfully from markdown block.")
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse extracted JSON string. Error: {e}")
                print(f"Malformed JSON string: {json_string}") # Print malformed string
                # If parsing fails, return an error or the raw output for debugging
                return jsonify({"error": f"Failed to parse Gemini's JSON output: {e}", "raw_gemini_output": response.text}), 500
        else:
            print("Warning: No JSON markdown block found in Gemini response. Returning raw text output.")
            # If no JSON block is found, return an error or the raw output for debugging
            return jsonify({"error": "Gemini did not return expected JSON markdown block.", "raw_gemini_output": response.text}), 500

        return jsonify(results) # This is the line that sends the response to the client

    except Exception as e:
        print(f"Error generating combined analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error generating analysis: {str(e)}. Please check the API key or try again."}), 500

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

if __name__ == '__main__':
    # When running locally, Flask defaults to http://127.0.0.1:5000
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) # Explicitly set port for clarity
