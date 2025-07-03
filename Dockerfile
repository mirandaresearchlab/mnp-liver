# Use an official Jupyter base image with Python
FROM jupyter/minimal-notebook:latest

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install uv (faster dependency manager)
RUN pip install uv

WORKDIR /app
COPY . /app

# Install dependencies from lockfile
RUN uv sync

# Expose the port Jupyter Notebook runs on
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]