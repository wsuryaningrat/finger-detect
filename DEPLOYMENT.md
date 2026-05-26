# 🚀 Deployment Guide - Streamlit Community Cloud

Panduan lengkap untuk mendeploy aplikasi Fingerprint Verification ke Streamlit Community Cloud.

## ✅ Checklist Pre-Deployment

- [x] `streamlit_app.py` ada di root directory
- [x] `requirements.txt` lengkap dengan semua dependencies
- [x] `.streamlit/config.toml` sudah dikonfigurasi
- [x] Model file (`models/best_model.pth`) tersedia
- [x] Source files (`src/`) terstruktur dengan baik
- [x] `.gitignore` sudah dibuat

## 📋 Step-by-Step Deployment

### 1. Persiapan Repository

```bash
# Pastikan semua file sudah di-commit
git status

# Jika ada untracked files
git add -A
git commit -m "Prepare for Streamlit Cloud deployment"
```

### 2. Push ke GitHub

```bash
# Push ke repository
git push origin main

# Verifikasi file di GitHub
# - streamlit_app.py (di root)
# - requirements.txt
# - .streamlit/config.toml
# - src/ folder dengan semua modules
# - models/best_model.pth
# - README.md
```

### 3. Deploy ke Streamlit Cloud

#### Metode A: Direct dari GitHub (Recommended)

1. Buka https://share.streamlit.io
2. Login dengan akun GitHub Anda
3. Click "New app"
4. Pilih repository: `finger-detect`
5. Pilih branch: `main`
6. Pilih file: `streamlit_app.py`
7. Click "Deploy"

#### Metode B: Manual via Streamlit Cloud Dashboard

1. Go to https://share.streamlit.io/create
2. Paste GitHub URL: `https://github.com/YOUR-USERNAME/finger-detect`
3. Enter main file path: `streamlit_app.py`
4. Click "Deploy"

### 4. Monitor Deployment

Streamlit Cloud akan:
- Clone repository
- Install dependencies dari `requirements.txt`
- Run `streamlit run streamlit_app.py`
- Deploy aplikasi

Proses biasanya selesai dalam 2-5 menit.

## 🔧 Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'src'"

**Solusi:**
- Pastikan path di `streamlit_app.py` menggunakan `ROOT = Path(__file__).resolve().parent`
- Verifikasi struktur folder di GitHub

### ❌ "FileNotFoundError: models/best_model.pth"

**Solusi:**
- Model file harus di-commit ke GitHub (gunakan Git LFS jika > 100MB)
- Pastikan path relatif benar

```bash
# Check file size
ls -lh models/best_model.pth

# Jika > 100MB, gunakan Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track model with Git LFS"
git push
```

### ❌ "RuntimeError: CUDA not available"

**Solusi:**
- Streamlit Cloud menggunakan CPU saja
- Code sudah otomatis fallback ke CPU
- Device akan menjadi "cpu" (normal)

### ❌ "Streamlit App stopped"

**Solusi:**
- Check logs di Streamlit Cloud dashboard
- Biasanya karena memory limitation
- Optimize memory usage:

```python
# Di streamlit_app.py
import streamlit as st

# Set max upload size
st.set_page_config(...)

# Clear cache periodically
@st.cache_data(ttl=3600)  # Cache expires after 1 hour
def preprocess_bytes(file_bytes):
    ...
```

## 📦 Requirements.txt Best Practices

```txt
# Format: package==version
streamlit==1.57.0
torch==2.12.0
torchvision==0.27.0
opencv-python-headless==4.13.0.92  # Use headless untuk server
numpy==1.24.2
scikit-image==0.20.0
scikit-learn==1.8.0
pillow==10.3.0
```

**Notes:**
- Gunakan `opencv-python-headless` (bukan `opencv-python`) di server
- Specify exact versions untuk reproducibility
- Regular update dependencies dengan: `pip freeze > requirements.txt`

## 🔒 Secrets Management

Jika ada API keys atau credentials:

1. **Create `.streamlit/secrets.toml`** (local only, jangan commit!)
2. **Di Streamlit Cloud Dashboard:**
   - Settings → Secrets → Paste secrets
3. **Access di code:**
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```

## 📊 Monitoring & Analytics

Streamlit Cloud provides:
- App health metrics
- Error logs
- Memory/CPU usage
- Visitor analytics

Akses via: https://share.streamlit.io → Your App → Settings

## 🔄 Continuous Updates

Untuk update aplikasi:

```bash
# 1. Buat changes locally
git add .
git commit -m "Update app"
git push origin main

# 2. Streamlit Cloud otomatis detect push
# 3. App akan redeploy otomatis dalam 1-2 menit
```

## 🎯 Performance Tips

1. **Use Caching**
   ```python
   @st.cache_resource  # Cache objects across reruns
   def load_model():
       ...

   @st.cache_data  # Cache data
   def preprocess_bytes(file_bytes):
       ...
   ```

2. **Optimize Model**
   - Gunakan model quantization untuk faster inference
   - Consider model distillation untuk smaller size

3. **Image Optimization**
   - Compress images sebelum upload
   - Limit upload size di config

## 📈 Scaling

Untuk traffic tinggi:
- Upgrade ke Streamlit Teams (paid)
- Implement load balancing
- Cache results aggressively

## ✨ Final Checklist

- [x] All files committed to GitHub
- [x] `streamlit_app.py` di root
- [x] `requirements.txt` lengkap
- [x] Model file termasuk
- [x] `.streamlit/config.toml` sudah setup
- [x] `.gitignore` configured
- [x] README.md tersedia
- [x] App tested locally
- [x] Deployed successfully
- [x] Monitoring enabled

## 🆘 Butuh Bantuan?

- Streamlit Docs: https://docs.streamlit.io
- Community Forum: https://discuss.streamlit.io
- GitHub Issues: https://github.com/streamlit/streamlit/issues

---

**Happy Deploying! 🚀**
