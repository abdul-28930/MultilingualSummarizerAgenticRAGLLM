Multilingual Speech Recognition and Analysis

This project uses fine-tuned Whisper models and DeepSeek for multilingual speech recognition and analysis.

## Prerequisites

1. Install Docker and Docker Compose on your system:
   - [Docker Desktop for Windows/Mac](https://www.docker.com/products/docker-desktop/)
   - For Linux: [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Get required API keys:
   - DeepSeek API key from [DeepSeek Platform](https://platform.deepseek.com)
   - Supabase credentials (if using Supabase)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a `.env` file in the project root:
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
DB_PASSWORD=your_database_password
```

3. Start the application:
```bash
docker-compose up -d
```

4. Access the application:
- Open your browser and go to: http://localhost:8501

## Project Structure

```
.
├── Deepseek/                  # Main application code
├── FineTune/                  # Fine-tuned Whisper models
├── docker-compose.yml         # Docker services configuration
├── Dockerfile                 # Main application container
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables (create this)
```

## Stopping the Application

To stop the application:
```bash
docker-compose down
```

## Troubleshooting

1. If you see permission errors:
```bash
sudo chown -R $USER:$USER .
```

2. If the models aren't loading:
- Ensure the models are in the correct directory
- Check the model paths in `initialize.py`

3. For database connection issues:
- Verify your database credentials in `.env`
- Check if the database container is running: `docker ps` 