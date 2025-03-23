# Integration Guide: Connecting React Frontend with Flask Backend

This guide explains how the React frontend and Flask backend have been integrated in the Multilingual Summarizer project.

## Architecture Overview

The integration follows a client-server architecture:

1. **Frontend**: React application built with Vite, located in `frontend/multilingual-frontend-main/`
2. **Backend**: Flask API that exposes endpoints for the frontend, located in `api.py`

## How the Integration Works

### Backend (Flask API)

The Flask API (`api.py`) provides several endpoints:

- `/api/health` - Health check endpoint
- `/api/transcribe` - Transcribes audio files
- `/api/summarize` - Generates extractive or abstractive summaries
- `/api/translate` - Translates text to different languages
- `/api/query` - Allows querying summaries

The API reuses the core functionality from the original Streamlit app but exposes it through RESTful endpoints.

### Frontend (React)

The React frontend has been updated to communicate with the Flask API:

1. **API Service** (`src/services/api.js`) - Contains functions for making API calls
2. **Components** - Updated to use the API service instead of placeholder data
3. **API Status Indicator** - Shows the connection status with the backend

## Running the Integrated Application

### Step 1: Start the Flask Backend

```bash
# From the project root directory
pip install -r requirements.txt
python api.py
```

The Flask API will start on http://localhost:5000

### Step 2: Start the React Frontend

```bash
# From the project root directory
cd frontend/multilingual-frontend-main
npm install
npm run dev
```

The React app will start on http://localhost:5173

### Step 3: Verify the Connection

1. Open the React app in your browser
2. Check the API status indicator in the bottom right corner
3. If it shows "API Connected" in green, the integration is working

## Troubleshooting

### Backend Issues

- **Missing dependencies**: Ensure all Python dependencies are installed
- **Port conflicts**: Make sure port 5000 is not in use by another application
- **API key issues**: Verify your OpenAI API key is correctly set in the `.env` file

### Frontend Issues

- **API connection errors**: Check if the backend is running and accessible
- **CORS issues**: The backend has CORS enabled, but you might need to adjust settings
- **Dependency issues**: Run `npm install` to ensure all dependencies are installed

## Data Flow

1. User uploads an audio file in the React UI
2. Frontend sends the file to the `/api/transcribe` endpoint
3. Backend processes the file and returns the transcription
4. User requests summarization
5. Frontend sends the text to the `/api/summarize` endpoint
6. Backend generates the summary and returns it with metrics
7. User can translate or query the summary through additional API calls

## Security Considerations

- The API doesn't implement authentication (consider adding this for production)
- API keys are stored in the backend `.env` file (never expose them to the frontend)
- File uploads should be validated and sanitized

## Future Improvements

- Add user authentication
- Implement WebSockets for real-time updates
- Add caching for better performance
- Implement proper error handling and retry mechanisms
