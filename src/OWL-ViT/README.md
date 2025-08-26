# OWL-ViT on Simulated Jetson Orin (ARM64)

This repository demonstrates how to run OWL-ViT (Object detection with vision-language models from HuggingFace)
inside a Docker container that simulates an **ARM64-based Jetson Orin** environment.  
The container uses the `l4t-ml` base image and supports testing code compatibility on ARM without real hardware.

---

## 📦 Project Structure

| File | Purpose |
|------|---------|
| `Dockerfile.dev-x86` | Development image for regular x86 systems |
| `Dockerfile.test-arm64` | ARM64-compatible image using Jetson's base image |
| `docker-compose.yml` | Optional compose file to manage builds |
| `run_owl.py` | Sample inference script for OWL-ViT |
| `image.jpg` | Sample input image |
| `requirements.txt` | Python dependencies |

---

## 🚀 Getting Started

### 1. Build the ARM64 Docker image

```bash
docker buildx build --platform=linux/arm64 -f Dockerfile.test-arm64 -t owl-arm64 .
```

```bash
docker run -it --rm \
  --platform=linux/arm64 \
  --memory=8g \
  -v $PWD:/workspace \
  -w /workspace \
  owl-arm64 python3 run_owl.py
```