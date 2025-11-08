# Fly.io Deployment Guide

## Prerequisites
1. Install Fly.io CLI:
```bash
curl -L https://fly.io/install.sh | sh
```

2. Login to Fly.io:
```bash
flyctl auth login
```

## Deployment Steps

### First Time Deployment

1. **Launch the app:**
```bash
flyctl launch
```
- Choose app name or press Enter to use "vimd-medical-ai"
- Select region (cdg = Paris recommended for MENA)
- Don't setup Postgres database
- Don't deploy yet

2. **Set memory/resources (important for ML models):**
```bash
flyctl scale memory 1024
```

3. **Deploy:**
```bash
flyctl deploy
```

### Subsequent Deployments

Just run:
```bash
flyctl deploy
```

## Useful Commands

### View logs
```bash
flyctl logs
```

### Check app status
```bash
flyctl status
```

### SSH into the container
```bash
flyctl ssh console
```

### Scale resources (if needed)
```bash
flyctl scale memory 2048  # 2GB RAM
flyctl scale vm shared-cpu-2x  # More CPU
```

### Open app in browser
```bash
flyctl open
```

### Check app info
```bash
flyctl info
```

## Important Notes

- **Model Size**: Your app includes ~489MB of ML models
- **Memory**: Configured for 1GB RAM (minimum for TensorFlow)
- **Region**: Default is Paris (cdg) - closest to MENA region
- **Auto-sleep**: App will sleep after inactivity to save costs
- **Cold Start**: First request after sleep takes ~40s (model loading)

## Troubleshooting

### Deployment fails
```bash
flyctl logs
```

### Out of memory
```bash
flyctl scale memory 2048
```

### Need to update config
Edit `fly.toml` then:
```bash
flyctl deploy
```

### Delete app
```bash
flyctl apps destroy vimd-medical-ai
```

## Cost Estimation

With Fly.io free tier:
- You get 3 shared-cpu-1x machines with 256MB RAM each (won't work for this app)
- Upgrade needed: ~$5-10/month for 1GB RAM machine
- Check pricing: https://fly.io/docs/about/pricing/
