# 🔍 Fingerprint Verification System

Siamese Neural Network untuk verifikasi fingerprint dengan analisis minutiae menggunakan dataset SOCOFing.

## 📋 Fitur Utama

- **Siamese Network (ResNet18)** - Ekstraksi feature dari pasangan fingerprint
- **Verifikasi Real-time** - Perhitungan euclidean distance untuk matching
- **Analisis Minutiae** - Visualisasi edge endings dan bifurcation points
- **ROC-AUC 0.9994** - Akurasi tinggi pada dataset validasi

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone <your-repo>
cd finger-detect

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# atau
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

### Deploy ke Streamlit Community Cloud

1. **Push ke GitHub**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Deploy via Streamlit Cloud**
   - Kunjungi https://share.streamlit.io
   - Click "New app"
   - Pilih repository, branch, dan file (`streamlit_app.py`)
   - Click "Deploy"

## 📦 Dependencies

- `streamlit` - Web framework
- `torch` & `torchvision` - Deep learning
- `opencv-python-headless` - Image processing
- `scikit-image` - Image analysis
- `numpy`, `pillow` - Data processing

## 📁 Project Structure

```
finger-detect/
├── streamlit_app.py          # Main app (Community Cloud friendly)
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── src/
│   ├── model.py            # Siamese Network architecture
│   ├── utils.py            # Utility functions
│   ├── minutiae.py         # Minutiae extraction
│   ├── preprocessing.py    # Image preprocessing
│   └── ...
├── models/
│   └── best_model.pth      # Pre-trained model
└── data/
    └── pairs_*.csv         # Training/validation data
```

## 🎯 Model Architecture

```
Siamese Network
├── Input: 224×224 grayscale image
├── Backbone: ResNet18 (11.2M parameters)
├── Embedding: 512-dimensional vector
├── Loss: Contrastive Loss
└── Similarity: Euclidean Distance
```

## 📊 Usage

1. **Upload** dua fingerprint dalam format JPG/PNG/BMP
2. **Click Verify** untuk memproses
3. **Lihat hasil** berupa similarity score dan decision
4. **Analisis minutiae** untuk explainability

## 🔬 Preprocessing Pipeline

1. **Grayscale Conversion** - Convert ke 8-bit grayscale
2. **Normalization** - Zero-mean standardization
3. **CLAHE** - Contrast Limited Adaptive Histogram Equalization
4. **Resize** - Resize ke 224×224 pixels
5. **Tensor Transform** - Convert ke normalized PyTorch tensor

## 📈 Performance

- **Training Data**: 6000 image pairs (SOCOFing)
- **Validation ROC-AUC**: 0.9994
- **Inference Time**: ~30ms per pair (CPU)
- **Threshold**: 0.2486

## 👥 Contributors

**Designed & Developed by WHS • Telkom University**

## 📝 License

MIT License - feel free to use and modify

## 🔗 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Documentation](https://pytorch.org/docs)
- [SOCOFing Dataset](https://github.com/growingpains/SOCOFing)
