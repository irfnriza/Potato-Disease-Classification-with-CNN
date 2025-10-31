# Autoencoder untuk Deteksi Penyakit Tanaman Kentang

Proyek ini menggunakan **Autoencoder berbasis PyTorch** untuk mendeteksi anomali (penyakit) pada daun kentang dengan pendekatan unsupervised learning. Model dilatih hanya pada gambar daun sehat, dan mendeteksi penyakit berdasarkan reconstruction error.

## 📋 Deskripsi Proyek

Model Autoencoder belajar merekonstruksi gambar daun kentang yang sehat. Ketika diberikan gambar daun berpenyakit, model akan menghasilkan reconstruction error yang tinggi karena pola penyakit tidak pernah dipelajari selama training.

### Dataset
- **Training**: Hanya gambar `Potato___healthy`
- **Testing**: Semua kelas (Healthy, Early_blight, Late_blight)
- **Augmentasi**: Random horizontal flip, rotation ±15°, resize 128×128

## 🏗️ Arsitektur Model

### Encoder
```
Input (3×128×128) 
  → Conv2D(3→16, stride=2) + ReLU → (16×64×64)
  → Conv2D(16→32, stride=2) + ReLU → (32×32×32)
  → Conv2D(32→64, stride=2) + ReLU → (64×16×16)
  → Conv2D(64→128, stride=2) + ReLU → (128×8×8)
```

### Decoder
```
Latent (128×8×8)
  → ConvTranspose2D(128→64) + ReLU → (64×16×16)
  → ConvTranspose2D(64→32) + ReLU → (32×32×32)
  → ConvTranspose2D(32→16) + ReLU → (16×64×64)
  → ConvTranspose2D(16→3) + Sigmoid → (3×128×128)
```

## 🚀 Fitur Utama

1. **Data Loading Modular**
   - Augmentasi on-the-fly untuk training
   - Split train/validation (80/20)
   - Support untuk multiple classes

2. **Training Components**
   - Loss: MSELoss
   - Optimizer: Adam (lr=1e-3)
   - Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
   - Early Stopping (patience=7)
   - Model Checkpoint (simpan model terbaik)

3. **Evaluasi & Visualisasi**
   - Reconstruction error per kelas
   - Histogram dan boxplot distribusi error
   - Perbandingan visual: original vs reconstructed vs error map
   - Learning curve dan learning rate schedule
   - Anomaly detection dengan threshold sederhana

4. **Reproducibility**
   - Random seed = 42 untuk semua komponen
   - Deterministic CUDA operations

## 📊 Output Files

Setelah training selesai, file berikut akan dibuat di folder `model/`:

- `best_autoencoder.pth` - Model terbaik (val_loss minimum)
- `training_history.png` - Learning curve dan LR schedule
- `error_distribution.png` - Histogram dan boxplot error
- `reconstruction_examples.png` - Contoh rekonstruksi per kelas
- `low_vs_high_error.png` - Perbandingan daun sehat vs berpenyakit

## 🔧 Requirements

```bash
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
tqdm
Pillow
```

## 📝 Cara Menggunakan

### 1. Install Dependencies
```powershell
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm Pillow
```

### 2. Struktur Folder
Pastikan struktur folder seperti ini:
```
Autoencoder/
├── dataset/
│   ├── Potato___Early_blight/
│   ├── Potato___healthy/
│   └── Potato___Late_blight/
├── model/                    # Akan dibuat otomatis
├── notebook/
│   └── notebook.ipynb
└── README.md
```

### 3. Jalankan Notebook
1. Buka `notebook/notebook.ipynb` di Jupyter atau VS Code
2. Jalankan semua cell secara berurutan
3. Model akan dilatih selama maksimal 50 epoch (atau sampai early stopping)
4. Output akan tersimpan di folder `model/`

### 4. Prediksi Gambar Baru (Optional)
Gunakan fungsi `predict_single_image()` di cell terakhir:

```python
result = predict_single_image(
    r"path/to/your/image.jpg",
    model, 
    threshold, 
    device
)
print(result)
```

## 📈 Expected Results

Model yang baik akan menunjukkan:
- **Daun sehat**: Reconstruction error rendah (< threshold)
- **Daun berpenyakit**: Reconstruction error tinggi (> threshold)
- **Threshold**: Biasanya dihitung sebagai `mean + 2×std` dari error daun sehat

### Metrik Evaluasi
- Accuracy: Seberapa akurat klasifikasi normal vs anomaly
- Precision: Dari yang diprediksi penyakit, berapa yang benar
- Recall: Dari semua penyakit, berapa yang terdeteksi
- F1-Score: Harmonic mean dari precision dan recall

## 🎯 Interpretasi Hasil

### Learning Curve
- **Train loss menurun**: Model belajar merekonstruksi gambar sehat
- **Val loss stabil**: Model tidak overfit
- **Gap kecil**: Generalisasi baik

### Error Distribution
- **Healthy**: Error rendah, distribusi sempit
- **Diseased**: Error tinggi, distribusi lebih lebar
- **Threshold**: Pemisah yang baik antara kedua kelompok

### Visualisasi Rekonstruksi
- **Healthy**: Original ≈ Reconstructed (error map gelap)
- **Diseased**: Original ≠ Reconstructed (error map terang pada area penyakit)

## 🔬 Hyperparameter Tuning

Jika hasil kurang memuaskan, coba:

1. **Learning Rate**: 1e-4 atau 5e-4 (lebih stabil tapi lambat)
2. **Batch Size**: 16 (lebih stabil) atau 64 (lebih cepat)
3. **Architecture**: Tambah layer atau channels
4. **Augmentation**: Tambah color jitter, brightness
5. **Threshold**: Gunakan percentile atau ROC curve

## 📚 Referensi

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Autoencoder Tutorial](https://pytorch.org/tutorials/beginner/introyt/autoencoderyt.html)
- [Anomaly Detection with Autoencoders](https://arxiv.org/abs/1901.03407)

## 👤 Author

Proyek ini dibuat sebagai implementasi Autoencoder untuk deteksi anomali pada tanaman kentang.

## 📄 License

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.

---

**Happy Coding! 🚀**
