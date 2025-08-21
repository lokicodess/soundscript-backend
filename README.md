
# SoundScript - Server-Side 

This directory contains the Flask backend application for SoundScript . It provides the API endpoints for audio transcription using OpenAI's Whisper model and conversation summarization using Google's Gemini API.

## Technologies Used

**Python**: The core programming language.

**Flask**: A lightweight web framework for building the API.

**Whisper (openai-whisper)**: OpenAI's powerful open-source speech-to-text model for robust transcription.

**Google Generative AI (google-generativeai)**: Python client library for interacting with Google's Gemini models.

**python-dotenv**: For managing environment variables.

**Flask-CORS**: A Flask extension to handle CORS headers.

## Setup and Installation

Follow these steps to get the SoundScript backend running on your local machine.

### Navigate to the Server Directory:

```bash
cd mvp/SoundScript/Server/
```

### Create and Activate a Virtual Environment:

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
```

### Install Python Dependencies:

```bash
pip install .
```
This will install Flask, Whisper, Google Generative AI, python-dotenv, and Flask-CORS.

### Download Whisper Model:

The whisper library requires a local model file. Download the base.pt model (or a larger one like small.pt or medium.pt for better accuracy, at the cost of disk space and processing time) from the official Whisper GitHub repository or by running a command like:


```bash
python -c "import whisper; whisper.load_model('base/small/medium/large-v3')"
```

Crucially, ensure the model_path variable in your app.py points to the correct local path where you have saved the .pt model file. For example:

```bash
model_path = "C:/WhisperModels/whisper/base.pt" # Example Windows path

model_path = "/home/user/.cache/whisper/base.pt" # Example Linux path
```

### Configure Gemini API Key:

Create a file named .env in this Server/ directory (same level as app.py and requirements.txt) and add your Google Gemini API key:

```bash
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

You can obtain a Gemini API key from the Google AI Studio.

## Running the Backend Server

Once all dependencies are installed, the API key is configured, add this path

```bash
cd mvp/SoundScript/Server/
```

and then run the backend server

```bash
python app.py
```

The server will typically run on http://127.0.0.1:5000. Keep this terminal open while using the frontend.

## API Endpoints

The backend provides the following endpoints:

**POST /transcribe**

Description: Transcribes an uploaded audio file into text.

Request: multipart/form-data with an audio file.

Response: application/json containing {"transcribedText": "..."} or {"error": "..."}.

**POST /summarize**

Description: Generates a concise summary from provided transcribed text.

Request: application/json with {"text": "Your transcribed text here"}.

Response: application/json containing {"summary": "..."} or {"error": "..."}.
