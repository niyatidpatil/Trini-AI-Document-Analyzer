FROM python:3.11
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0



# # 1. Use a lightweight Python Linux image
# FROM python:3.11

# # 2. Set the working directory inside the container
# WORKDIR /app

# # 3. Install system tools needed for PDF parsing (PyMuPDF dependencies)
# # We removed 'software-properties-common' as it was causing the error.
# # We added --no-install-recommends to keep the image small.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # 4. Copy your requirements file
# COPY requirements.txt .

# # 5. Install Python libraries
# # Added --upgrade pip to ensure smooth installation
# RUN pip3 install --upgrade pip && \
#     pip3 install --no-cache-dir -r requirements.txt

# # 6. Copy your app code
# COPY . .

# # 7. Tell Docker we are using port 8080
# EXPOSE 8080

# # 8. The command to start Trini AI
# CMD streamlit run app.py --server.port 8080 --server.address 0.0.0.0

