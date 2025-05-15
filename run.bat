@echo off
echo Stopping any existing containers...
docker stop quantum-api 2>nul
docker rm quantum-api 2>nul

echo.
echo Building and running the Quantum Optimization API...

REM Build the Docker image
docker build -t quantum-optimization-api .

REM Run the Docker container
docker run -d -p 8000:8000 --name quantum-api quantum-optimization-api

echo.
echo API is now running at http://localhost:8000
echo.
echo To run the client with your Excel file, use:
echo python client.py path/to/your/excel/file.xlsx
echo.
echo To stop the API, run:
echo docker stop quantum-api
echo docker rm quantum-api 