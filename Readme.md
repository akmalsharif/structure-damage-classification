# Building Damage Grade Classification

Proyek ini mengklasifikasikan tingkat kerusakan bangunan berdasarkan informasi
bangunan dan lokasi. Target `damage_grade` berisi lima kelas: `Grade 1` sampai
`Grade 5`.

## Struktur Data

- `data/train.csv`: data pelatihan, termasuk target.
- `data/test.csv`: data yang akan diprediksi.
- `data/sample_submission.csv`: contoh format dan label submission.
- `data/metadata.csv`: deskripsi kolom pada dataset.

## Mengunduh Data dari Kaggle

Data tersedia di halaman [Structural Damage Classification di Kaggle](https://www.kaggle.com/competitions/structural-damage-classification/data).
Login ke Kaggle dan setujui aturan kompetisi sebelum mengunduh data.

Pasang Kaggle CLI di virtual environment proyek:

```powershell
python -m pip install kaggle
```

Autentikasi satu kali dengan API token Kaggle:

1. Buka pengaturan akun Kaggle, bagian **API**, lalu buat/download token API.
2. Simpan file `kaggle.json` di `%USERPROFILE%\.kaggle\kaggle.json`.
   Buat folder `.kaggle` jika belum ada.
3. Jangan masukkan atau membagikan file token tersebut ke repository.

Dari root proyek, unduh dan ekstrak arsip:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
kaggle competitions download -c structural-damage-classification -p data
Expand-Archive -Path data\structural-damage-classification.zip -DestinationPath data -Force
```

Pastikan `data\train.csv`, `data\test.csv`, dan
`data\sample_submission.csv` tersedia. File data dan arsip ZIP di folder `data`
diabaikan oleh Git.

Training memakai `scripts/feature_engineering.py` dan
`scripts/preprocess.py` untuk menyelaraskan fitur train dan test. Kolom ID
`building_id` dan target tidak dipakai sebagai fitur; ID tetap disertakan pada
file submission.

## Model dan Evaluasi

Script training utama menggunakan LightGBM multiclass dengan fitur kategorikal
native. Hyperparameter dituning menggunakan Optuna TPE untuk memaksimalkan
Macro F1 pada stratified 3-fold cross-validation. Tuning memakai maksimal
100.000 baris dan 30 percobaan; model terbaik kemudian dilatih pada seluruh
bagian training dari split 80/20.

Data validasi ditahan terpisah dengan stratified split (20%) untuk laporan
Macro F1 dan classification report. Distribusi target tidak seimbang, sehingga
tuning juga membandingkan `class_weight=None` dan `class_weight="balanced"`.
Skor validasi adalah estimasi lokal, bukan jaminan skor leaderboard.

Notebook `notebooks/02_modeling.ipynb` berisi eksperimen CatBoost terpisah;
script di bawah adalah jalur training LightGBM utama.

## Persiapan Environment

Dari root proyek, buat dan aktifkan virtual environment, lalu pasang dependency:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Pilih environment `.venv` sebagai interpreter/kernel Python di VS Code.

## Menjalankan

Jalankan perintah berikut dari root proyek:

```bash
python -m scripts.train
python -m scripts.predict
```

Training menjalankan 30 percobaan Optuna dengan 3-fold CV; durasinya bergantung
pada perangkat. Prediksi memakai model yang disimpan oleh langkah training.

## Output

- Model dan metadata training: `models/pipeline_model_lightgbm.pkl`
- Confusion matrix validasi: `reports/confusion_matrix.csv`
- Submission: `submissions/submission.csv`

Submission memiliki kolom `building_id` dan `damage_grade`. Sebelum disimpan,
script prediksi memeriksa jumlah baris, urutan ID, kolom, nilai kosong, dan label
terhadap `sample_submission.csv`.

Contoh format:

```csv
building_id,damage_grade
122207000141, 3
284909000871, 1
241904000231, 5
```
