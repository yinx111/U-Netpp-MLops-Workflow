
## Web App + Cloudflare Tunnel (Docker Compose)
This repo includes:
- `app/app.py`: FastAPI web app (upload/download UI), directly runs `scripts/inference.py` in the same container.
- `requirements_app.txt`: app + inference runtime dependencies.
- `docker-compose.yml`: `app` + `cloudflared` services.

### 1) Configure env vars
Reference `.env.app.example` and add these vars into your existing `.env`:
- `CLOUDFLARED_TOKEN`
- `APP_HOST`
- `APP_PORT`
- `MAX_UPLOAD_MB`

### 2) Build and start
```bash
docker compose up -d --build
```

Local app URL:
```bash
http://127.0.0.1:8000
```

### 3) Cloudflare domain binding
In Cloudflare Zero Trust:
1. Create a tunnel and get token.
2. Add a Public Hostname and point service to `http://app:8000`.
3. Ensure DNS record is proxied by Cloudflare.

## 📜 Data Usage and Attribution

This dataset was created using open-source satellite imagery from the **U.S. Department of Agriculture National Agriculture Imagery Program (NAIP)**.

If you use this dataset in your research or project, please cite or acknowledge:

> Source imagery: USDA NAIP  
> Processed and labeled dataset: yinx111 (2025), https://github.com/yinx111/U-Net-Semantic-Segmentation-on-Multispectral-RGB-NIR-Imagery

**And the dataset will be continuously expanded with additional land-cover categories and samples in future updates.**


## Web App + Cloudflare Tunnel (Docker Compose)
This repo includes:
- `app/app.py`: FastAPI web app (upload/download UI), directly runs `scripts/inference.py` in the same container.
- `requirements_app.txt`: app + inference runtime dependencies.
- `docker-compose.yml`: `app` + `cloudflared` services.

### 1) Configure env vars
Reference `.env.app.example` and add these vars into your existing `.env`:
- `CLOUDFLARED_TOKEN`
- `APP_HOST`
- `APP_PORT`
- `MAX_UPLOAD_MB`

### 2) Build and start
```bash
docker compose up -d --build
```

Local app URL:
```bash
http://127.0.0.1:8000
```

### 3) Cloudflare domain binding
In Cloudflare Zero Trust:
1. Create a tunnel and get token.
2. Add a Public Hostname and point service to `http://app:8000`.
3. Ensure DNS record is proxied by Cloudflare.

## 📜 Data Usage and Attribution

This dataset was created using open-source satellite imagery from the **U.S. Department of Agriculture National Agriculture Imagery Program (NAIP)**.

If you use this dataset in your research or project, please cite or acknowledge:

> Source imagery: USDA NAIP  
> Processed and labeled dataset: yinx111 (2025), https://github.com/yinx111/U-Net-Semantic-Segmentation-on-Multispectral-RGB-NIR-Imagery

**And the dataset will be continuously expanded with additional land-cover categories and samples in future updates.**
