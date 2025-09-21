#this is the base image
FROM  python:3.9-slim

# creates a directory named /app and makes it our current location
WORKDIR /app

# Copy the Python script into the container
COPY DroneEnvironment.py .

COPY requirements.txt .
RUN pip install -r requirements.txt

#Copy the rest of the project files
COPY . .

# Specify the command to run on container start
CMD ["python", "DroneEnvironment.py"]
