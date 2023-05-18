FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Set a build argument for the default port number
ARG PORT=8000

# Set an environment variable for the port number
ENV PORT=$PORT

# Expose the port
EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT