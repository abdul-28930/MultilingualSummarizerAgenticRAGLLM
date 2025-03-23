Multilingual Speech Recognition and Analysis

This project uses fine-tuned Whisper models and DeepSeek for multilingual speech recognition and analysis.

## Prerequisites

1. Install Docker and Docker Compose on your system:
   - [Docker Desktop for Windows/Mac](https://www.docker.com/products/docker-desktop/)
   - For Linux: [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Get required API keys:
   - OpenAI API key for summarization and translation
   - Supabase credentials (if using Supabase)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
DB_PASSWORD=your_database_password
```

## Running the Application

### Option 1: Using Streamlit (Original UI)

Run the Streamlit application:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Using Flask API with React Frontend (New UI)

1. Start the Flask API backend:
```bash
pip install -r requirements.txt
python api.py
```

2. Start the React frontend:
```bash
cd frontend/multilingual-frontend-main
npm install
npm run dev
```

3. Access the application:
- Streamlit UI: http://localhost:8501
- React UI: http://localhost:5173

## Project Structure

```
.
├── app.py                     # Streamlit application
├── api.py                     # Flask API for frontend integration
├── summary_evaluation.py      # Summary evaluation utilities
├── extractive_summarizer.py   # Extractive summarization module
├── abstractive_summarizer.py  # Abstractive summarization module
├── frontend/                  # React frontend application
│   └── multilingual-frontend-main/
│       ├── src/               # React source code
│       ├── public/            # Public assets
│       └── package.json       # Frontend dependencies
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables (create this)
```

## Features

- Multilingual speech recognition (Arabic, Hindi)
- Extractive and abstractive summarization
- Summary translation to multiple languages
- Interactive querying of summaries
- Summary quality evaluation metrics

## Troubleshooting

1. If you see permission errors:
```bash
sudo chown -R $USER:$USER .
```

2. If the models aren't loading:
- Ensure the models are in the correct directory
- Check the model paths in the application

3. For API connection issues:
- Verify your API keys in `.env`
- Check if both backend and frontend are running
- Look for the API status indicator in the bottom right of the React UI