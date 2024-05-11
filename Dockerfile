# Use Python 3.11.5
FROM python:3.11.5

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.8.2

# Copy the poetry.lock and pyproject.toml files
COPY poetry.lock pyproject.toml /app/

# Install dependencies using Poetry
RUN poetry install --no-interaction --no-root

# Copy the rest of the application code
COPY . /app

# Expose port 8080
EXPOSE 8080

# Set PYTHONPATH to include /app directory
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Command to run the application
CMD ["poetry", "run", "python", "src/serve/api.py"]
