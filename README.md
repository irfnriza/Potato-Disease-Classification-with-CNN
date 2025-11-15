# CNN untuk Klasifikasi Penyakit Tanaman Kentang

Proyek ini menggunakan **Convolutional Neural Network (CNN) murni berbasis PyTorch** untuk mengklasifikasikan penyakit pada daun kentang ke dalam 3 kategori: **Healthy**, **Early Blight**, dan **Late Blight**.

## 📋 Deskripsi Proyek

Model CNN dilatih untuk mengenali pola visual dari daun kentang yang sehat dan berpenyakit menggunakan pendekatan supervised learning. Arsitektur CNN murni (tanpa pretrained model) dirancang khusus untuk task klasifikasi multi-kelas ini.

### Dataset
- **Training & Validation**: Semua kelas (80% training, 20% validation)
- **Testing**: Semua kelas untuk evaluasi akhir
- **Kelas**: 
  - Potato___Early_blight
  - Potato___healthy
  - Potato___Late_blight
- **Augmentasi**: Random horizontal flip, rotation ±15°, color jitter, normalisasi ImageNet

## 🏗️ Arsitektur Model

### CNN Architecture
```
Input (3×128×128) 

Block 1: 3 → 32
  → Conv2D(3→32, 3×3, padding=1) + BatchNorm + ReLU
  → Conv2D(32→32, 3×3, padding=1) + BatchNorm + ReLU
  → MaxPool2D(2×2) + Dropout(0.1) → (32×64×64)

Block 2: 32 → 64
  → Conv2D(32→64, 3×3, padding=1) + BatchNorm + ReLU
  → Conv2D(64→64, 3×3, padding=1) + BatchNorm + ReLU
  → MaxPool2D(2×2) + Dropout(0.2) → (64×32×32)

Block 3: 64 → 128
  → Conv2D(64→128, 3×3, padding=1) + BatchNorm + ReLU
  → Conv2D(128→128, 3×3, padding=1) + BatchNorm + ReLU
  → MaxPool2D(2×2) + Dropout(0.3) → (128×16×16)

Block 4: 128 → 256
  → Conv2D(128→256, 3×3, padding=1) + BatchNorm + ReLU
  → Conv2D(256→256, 3×3, padding=1) + BatchNorm + ReLU
  → MaxPool2D(2×2) + Dropout(0.4) → (256×8×8)

Global Average Pooling → (256)

Classifier:
  → Linear(256→128) + ReLU + Dropout(0.5)
  → Linear(128→3)
```

**Total Parameters**: ~600K parameters

## 🚀 Fitur Utama

1. **CNN Murni (Tanpa Pretrained)**
   - Arsitektur custom-built dari scratch
   - 4 convolutional blocks dengan BatchNorm
   - Global Average Pooling untuk feature aggregation
   - Dropout untuk regularisasi

2. **Data Loading & Augmentasi**
   - Augmentasi on-the-fly untuk training
   - Split train/validation (80/20)
   - Normalisasi menggunakan ImageNet statistics
   - Support untuk multiple classes

3. **Training Components**
   - Loss: CrossEntropyLoss (untuk klasifikasi multi-kelas)
   - Optimizer: Adam (lr=1e-3, weight_decay=1e-4)
   - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
   - Early Stopping (patience=10)
   - Model Checkpoint (simpan model terbaik berdasarkan val_loss)

4. **Evaluasi & Visualisasi Lengkap**
   - Confusion Matrix (counts & normalized)
   - Classification Report (precision, recall, F1-score)
   - Learning curves (loss & accuracy)
   - Visualisasi prediksi benar vs salah
   - Visualisasi prediksi per kelas
   - Per-class performance metrics

5. **Reproducibility**
   - Random seed = 42 untuk semua komponen
   - Deterministic CUDA operations

## 📊 Output Files

Setelah training selesai, file berikut akan dibuat di folder `model/`:

- `best_cnn_model.pth` - Model terbaik (val_loss minimum)
- `training_history.png` - Learning curve (loss & accuracy) dan LR schedule
- `confusion_matrix.png` - Confusion matrix (counts & normalized)
- `prediction_samples.png` - Contoh prediksi benar vs salah
- `predictions_per_class.png` - Contoh prediksi untuk setiap kelas

## 🔧 Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tqdm>=4.65.0
Pillow>=10.0.0
```

## 📝 Cara Menggunakan

### 1. Install Dependencies

Gunakan **Conda** untuk membuat environment baru dengan Python 3.12.3 dan menginstal semua dependencies dari `requirements.txt`.

#### 🧩 Buat dan Aktifkan Environment

```bash
conda create -n myenv python=3.12.3
conda activate myenv
pip install -r requirements.txt
```

### 2. Struktur Folder
Pastikan struktur folder seperti ini:
```
Simple-Autoendcoder/
├── dataset/
│   ├── Potato___Early_blight/
│   ├── Potato___healthy/
│   └── Potato___Late_blight/
├── model/                    # Akan dibuat otomatis
├── notebook/
│   └── notebook.ipynb
├── README.md
└── requirements.txt
```

### 3. Jalankan Notebook
1. Buka `notebook/notebook.ipynb` di Jupyter atau VS Code
2. Jalankan semua cell secara berurutan
3. Model akan dilatih selama maksimal 100 epoch (atau sampai early stopping)
4. Output akan tersimpan di folder `model/`

### 4. Prediksi Gambar Baru
Gunakan fungsi `predict_single_image()` di cell terakhir:

```python
result = predict_single_image(
    r"path/to/your/image.jpg",
    model, 
    class_names,
    device
)
print(result)
```

## 📈 Expected Results

Model yang baik akan menunjukkan:
- **Accuracy**: > 90% pada test set
- **Confusion Matrix**: Diagonal values tinggi (prediksi benar)
- **Learning Curve**: Konvergen tanpa overfitting (train & val loss menurun bersamaan)
- **Per-class Performance**: Precision, recall, dan F1-score tinggi untuk semua kelas

### Metrik Evaluasi
- **Accuracy**: Persentase prediksi yang benar secara keseluruhan
- **Precision**: Dari yang diprediksi sebagai kelas tertentu, berapa yang benar
- **Recall**: Dari semua instance kelas tertentu, berapa yang terdeteksi
- **F1-Score**: Harmonic mean dari precision dan recall

## 🎯 Interpretasi Hasil

### Learning Curve
- **Loss menurun**: Model belajar dengan baik
- **Train & Val loss gap kecil**: Tidak overfitting
- **Accuracy meningkat**: Model semakin baik mengklasifikasi

### Confusion Matrix
- **Diagonal tinggi**: Prediksi benar untuk masing-masing kelas
- **Off-diagonal rendah**: Kesalahan klasifikasi minimal
- **Normalized view**: Menunjukkan persentase akurasi per kelas

### Visualisasi Prediksi
- **Correct predictions**: Model confident dengan probabilitas tinggi
- **Wrong predictions**: Biasanya terjadi pada kasus ambiguous atau borderline
- **Per-class samples**: Menunjukkan konsistensi model dalam setiap kelas

## 🔬 Hyperparameter Tuning

Jika hasil kurang memuaskan, coba:

1. **Learning Rate**: 
   - Lebih rendah (5e-4, 1e-4) untuk training lebih stabil
   - Lebih tinggi (5e-3) untuk konvergensi lebih cepat

2. **Batch Size**: 
   - Lebih kecil (16, 8) untuk generalisasi lebih baik
   - Lebih besar (64, 128) untuk training lebih cepat

3. **Architecture**: 
   - Tambah conv layers atau channels untuk capacity lebih besar
   - Kurangi dropout jika underfitting
   - Tambah dropout jika overfitting

4. **Augmentation**: 
   - Tambah: RandomResizedCrop, RandomAffine, GaussianBlur
   - Sesuaikan intensity (brightness, contrast, saturation)

5. **Training Strategy**:
   - Increase epochs jika belum converge
   - Adjust scheduler patience dan factor
   - Try different optimizers (SGD with momentum, AdamW)

## 📚 Referensi

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CNN for Image Classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Deep Learning for Computer Vision](https://cs231n.github.io/)
- [Plant Disease Classification Research](https://arxiv.org/abs/1604.03169)

## 👤 Author

Proyek ini dibuat sebagai implementasi CNN murni untuk klasifikasi penyakit tanaman kentang.

## 📄 License

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.

---

**Happy Coding! 🚀**
