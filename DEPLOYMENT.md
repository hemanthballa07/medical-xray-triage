# Deployment Guide

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose (optional, for easier management)
- NVIDIA Docker (for GPU support, optional)

### Build Docker Image

```bash
docker build -t medical-xray-triage .
```

### Run Container

**CPU only:**
```bash
docker run -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  medical-xray-triage
```

**With GPU support:**
```bash
docker run --gpus all -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  medical-xray-triage
```

### Using Docker Compose

```bash
# Start service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Access UI

Once running, access the UI at: `http://localhost:8501`

## Cloud Deployment (AWS EC2)

### Setup EC2 Instance

1. Launch EC2 instance (recommended: g4dn.xlarge for GPU)
2. Install Docker:
   ```bash
   sudo yum update -y
   sudo yum install docker -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user
   ```

3. Install NVIDIA Docker (for GPU instances):
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

4. Clone repository:
   ```bash
   git clone <repository-url>
   cd pneumonia-project
   ```

5. Build and run:
   ```bash
   docker build -t medical-xray-triage .
   docker run --gpus all -p 8501:8501 medical-xray-triage
   ```

6. Configure security group to allow port 8501

### Performance Comparison

| Device | Inference Latency | Throughput |
|--------|------------------|------------|
| CPU    | ~50-100 ms       | ~10-20 img/s |
| GPU    | ~5-10 ms         | ~100-200 img/s |
| MPS    | ~10-20 ms        | ~50-100 img/s |

## Conda Pack Deployment

### Create Portable Environment

```bash
# Activate environment
conda activate medxray

# Install conda-pack
conda install conda-pack

# Pack environment
conda pack -n medxray -o medxray.tar.gz
```

### Deploy to Target Machine

```bash
# Extract on target machine
mkdir medxray_env
tar -xzf medxray.tar.gz -C medxray_env

# Activate
source medxray_env/bin/activate

# Run application
streamlit run ui/app.py
```

## Production Considerations

1. **Model Caching**: Models are cached in memory for faster inference
2. **GPU Detection**: System automatically detects and uses GPU if available
3. **Resource Monitoring**: UI displays real-time CPU/GPU/memory usage
4. **Scaling**: Use load balancer for multiple instances
5. **Security**: Add authentication for production deployment
6. **Monitoring**: Integrate with monitoring tools (Prometheus, Grafana)

## Troubleshooting

### GPU Not Detected
- Check NVIDIA drivers: `nvidia-smi`
- Verify Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Port Already in Use
- Change port: `streamlit run ui/app.py --server.port=8502`

### Memory Issues
- Reduce batch size in config
- Use CPU if GPU memory insufficient

