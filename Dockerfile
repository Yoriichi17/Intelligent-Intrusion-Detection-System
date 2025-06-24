# ---- FRONTEND STAGE ----
FROM node:20 AS frontend

WORKDIR /Dashboard/frontend
COPY Dashboard/frontend/package*.json ./
RUN npm install
COPY Dashboard/frontend/ ./
RUN npm run build

# ---- BACKEND STAGE ----
FROM python:3.10-slim AS backend

# Set env vars
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /Dashboard/backend

# Install Python dependencies
COPY Dashboard/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY Dashboard/backend/ ./

# Copy frontend build from frontend stage
COPY --from=frontend /Dashboard/frontend/build ./frontend_build

# Create shell script to run uvicorn + realtime_predict.py
RUN echo '#!/bin/bash\n' \
         'uvicorn main:app --host 0.0.0.0 --port 5000 --reload &\n' \
         'python realtime_predict.py' > start.sh && chmod +x start.sh

# Expose backend port
EXPOSE 5000

CMD ["./start.sh"]
