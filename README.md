
## Web App + Cloudflare Tunnel (Docker Compose)
This repo includes:
- `app/app.py`: FastAPI web app (upload/download UI), directly runs `scripts/inference.py` in the same container.
- `requirements_app.txt`: app + inference runtime dependencies.
- `docker-compose.yml`: `app` + `cloudflared` services.

### 1) Configure env vars (optional)
Environment loading is optional. If you need custom values, copy `.env.example` to `.env` first.
For app-specific variables, reference `.env.app.example` and add these vars into `.env`:
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

## App CD (GitHub Actions + GHCR + Remote Docker Compose)

Workflow file: `.github/workflows/cd-app.yml`

- Auto deploy to `staging`: on `push` to `dev/main/master`.
- Manual deploy to `staging`: `workflow_dispatch`.
- Container image is pushed to `ghcr.io/<owner>/<repo>/app:<tag>`.

### Required GitHub Environments

Create one environment:
- `staging`

### Required secrets

`staging` environment:
- `STAGING_SSH_HOST`
- `STAGING_SSH_USER`
- `STAGING_SSH_KEY`
- `STAGING_DEPLOY_PATH` (absolute path on remote host)
- `STAGING_GHCR_USERNAME`
- `STAGING_GHCR_TOKEN` (PAT with `read:packages`)
- `CLOUDFLARED_TOKEN`

### Optional environment variables

Set in the `staging` GitHub environment `Variables`:
- `APP_HOST`
- `APP_PORT`
- `MAX_UPLOAD_MB`
- `INFERENCE_TIMEOUT_SECONDS`
- `CHECKPOINT_PATH`
- `FORCE_CPU`
- `CUDA_VISIBLE_DEVICES`

Deployment compose file used by CD:
- `deploy/docker-compose.cd.yml`

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

### 1) Configure env vars (optional)
Environment loading is optional. If you need custom values, copy `.env.example` to `.env` first.
For app-specific variables, reference `.env.app.example` and add these vars into `.env`:
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

# test