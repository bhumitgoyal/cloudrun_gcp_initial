#!/bin/bash

# Load environment variables if .env exists
if [ -f .env ]; then
  echo "Loading .env file..."
  export $(grep -v '^#' .env | xargs)
fi

PORT=${PORT:-8080}

echo "Starting GoHappy Club Chatbot on port $PORT..."
python main.py &
APP_PID=$!

echo "Starting ngrok to expose port $PORT to the internet..."
ngrok http $PORT

# Clean up the background running app when ngrok exits (Ctrl+C)
kill $APP_PID
echo "Server and ngrok stopped."
