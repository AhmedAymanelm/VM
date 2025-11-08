# VIMD Medical AI Application - Docker Setup

## Quick Start

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8501`

### Using Docker CLI
```bash
# Build the image
docker build -t vimd-app .

# Run the container
docker run -p 8501:8501 --name vimd-medical-ai vimd-app
```

## Docker Commands

### Build the image
```bash
docker-compose build
```

### Start the application
```bash
docker-compose up
```

### Start in detached mode (background)
```bash
docker-compose up -d
```

### Stop the application
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f
```

### Rebuild and restart
```bash
docker-compose up --build --force-recreate
```

## Configuration

- **Port**: The application runs on port 8501
- **Health Check**: Configured to check `/_stcore/health` endpoint
- **Restart Policy**: Container will automatically restart unless stopped manually

## Models Included

The Docker image includes all required models:
- Brain Tumor Classification Model (242MB)
- YOLO Brain Tumor Detection Model (5.2MB)
- Pancreas Tumor Classification Model (242MB)

Total size: ~489MB of models

## Requirements

- Docker Engine 20.10+
- Docker Compose 1.29+ (if using docker-compose)
- At least 2GB of free disk space for the image

## Troubleshooting

### Container fails to start
Check logs:
```bash
docker-compose logs
```

### Port already in use
Change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Health check failing
Wait 40 seconds for the application to fully start (configured start_period)
