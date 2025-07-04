FROM jupyter/minimal-notebook:latest
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
WORKDIR /home/jovyan/work
COPY mnp_analysis.ipynb .
COPY requirements.txt .
# Install system dependencies
USER root
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
USER jovyan
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]