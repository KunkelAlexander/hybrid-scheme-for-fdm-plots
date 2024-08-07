# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install any needed packages
RUN pip install numpy matplotlib pandas

# Copy the current directory contents into the container at /app
COPY . /app

# Command to run the script
CMD ["python", "filament.py"]