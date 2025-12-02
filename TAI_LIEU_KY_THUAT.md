# TÀI LIỆU KỸ THUẬT
# Hệ Thống Dự Đoán Thành Công Phim

Phiên bản: 1.0.0  
Cập nhật: 2025-11-20  

---

## MỤC LỤC

1. [Tổng Quan Hệ Thống](#1-tong-quan-he-thong)
2. [Kiến Trúc](#2-kien-truc)
3. [Tài Liệu Module](#3-tai-lieu-module)
4. [Hướng Dẫn Triển Khai Local](#4-huong-dan-trien-khai-local)
5. [Ví Dụ Sử Dụng](#5-vi-du-su-dung)
6. [API Reference](#6-api-reference)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Mục Đích

Hệ thống học máy đánh giá rủi ro đầu tư sản xuất phim dựa trên nhiều yếu tố: thành tích đạo diễn, uy tín diễn viên, lịch sử biên kịch và thể loại phim.

### 1.2 Tính Năng Chính

- Pipeline xử lý dữ liệu tự động
- 6 mô hình học máy (classification, regression, deep learning)
- Module inference production-ready
- REST API server với tài liệu tự động
- Hỗ trợ Docker

### 1.3 Công Nghệ

- Python 3.10+
- scikit-learn, XGBoost
- PyTorch, PyTorch Lightning (tùy chọn)
- FastAPI
- Pandas, NumPy

---

## 2. KIẾN TRÚC

### 2.1 Luồng Dữ Liệu

```
Dữ liệu IMDb thô
    ↓
preprocess.py (Làm sạch + Thống kê)
    ↓
feature.py (Trích xuất đặc trưng)
    ↓
train.py (Huấn luyện 6 models)
    ↓
evaluate_models.py (Chọn model tốt nhất)
    ↓
Mô hình tốt nhất
    ↓
    ├─→ main.py (Phát triển)
    └─→ predict.py → api_server.py (Production)
```

### 2.2 Các Module

- **preprocess.py**: Xử lý dữ liệu thô
- **feature.py**: Kỹ thuật đặc trưng
- **train.py**: 6 mô hình ML
- **evaluate_models.py**: So sánh models
- **main.py**: CLI đầy đủ chức năng
- **predict.py**: Inference module (production)
- **api_server.py**: REST API server

---

## 3. TÀI LIỆU MODULE

### 3.1 preprocess.py

**Mục đích**: Xử lý và làm sạch dữ liệu IMDb.

#### Hàm chính: preprocess_pipeline

```python
def preprocess_pipeline(data_path: str) -> pd.DataFrame
```

**Tham số**:
- `data_path` (str): Đường dẫn thư mục chứa file TSV của IMDb

**Trả về**:
- `pd.DataFrame`: DataFrame đã xử lý với các cột:
  - Metadata phim: tconst, primaryTitle, startYear, runtimeMinutes
  - ID người: directors_nconst, writers_nconst, cast_nconst
  - Thống kê: director_mean_rating, director_total_films, writer_mean_rating, writer_total_films, cast_mean_rating, cast_total_films
  - Đánh giá: averageRating, numVotes
  - Nhãn: is_success, is_risky

**Ví dụ**:
```python
from preprocess import preprocess_pipeline
df = preprocess_pipeline('../data/')
```

---

### 3.2 feature.py

**Mục đích**: Kỹ thuật đặc trưng và mã hóa thể loại.

#### encode_genres

```python
def encode_genres(df: pd.DataFrame, genre_col: str = 'genre_list') -> pd.DataFrame
```

Chuyển danh sách thể loại thành one-hot encoding.

**Tham số**:
- `df`: DataFrame đầu vào
- `genre_col`: Tên cột chứa thể loại (mặc định: 'genre_list')

**Trả về**: DataFrame với cột mới cho mỗi thể loại (genre_Action, genre_Drama,...)

#### get_feature_target

```python
def get_feature_target(df: pd.DataFrame, target_type: str = 'rating') -> Tuple[pd.DataFrame, pd.Series/DataFrame]
```

Tách dữ liệu thành ma trận đặc trưng X và biến mục tiêu y.

**Tham số**:
- `df`: DataFrame đã xử lý
- `target_type`: Loại mục tiêu
  - `'rating'`: Regression đơn (averageRating)
  - `'votes'`: Regression đơn (numVotes)
  - `'rating_votes'`: Regression đa đầu ra
  - `'is_risky'`: Phân loại nhị phân

**Trả về**: (X, y)

---

### 3.3 train.py

**Mục đích**: Huấn luyện 6 mô hình học máy.

#### 6 Mô Hình

**Phân loại (is_risky)**:
1. Logistic Regression với GridSearchCV
2. Random Forest Classifier với GridSearchCV

**Regression (rating/votes)**:
3. Random Forest Regressor (multi-output)
4. XGBoost Regressor (multi-output)

**Deep Learning (hybrid)**:
5. SimplerDNN - MLP đơn giản, không dùng embeddings
6. HybridMovieModel - Với embeddings cho đạo diễn/diễn viên/biên kịch

#### train_is_risky_models

```python
def train_is_risky_models(data_path: str, save_dir: str = "models/is_risky") -> Dict[str, Any]
```

Huấn luyện 2 models phân loại.

**Trả về**: Dictionary với keys 'logreg' và 'rf'

#### train_rating_votes_models

```python
def train_rating_votes_models(data_path: str, save_dir: str = "models/rating_votes") -> Dict[str, Any]
```

Huấn luyện 2 models regression.

**Trả về**: Dictionary với keys 'rf_reg' và 'xgb_reg'

---

### 3.4 main.py

**Mục đích**: Giao diện CLI thống nhất cho tất cả thao tác.

#### 4 Chế Độ Hoạt Động

**1. evaluate**: Đánh giá tất cả models, chọn tốt nhất
```bash
python main.py --mode evaluate --data-path ../data/
```

**2. train**: Huấn luyện model cụ thể
```bash
python main.py --mode train --model-type classification --data-path ../data/
```

**3. predict**: Dự đoán hàng loạt từ file CSV
```bash
python main.py --mode predict --model-path model.pkl --input-file test.csv
```

**4. interactive**: Dự đoán tương tác từng phim
```bash
python main.py --mode interactive --model-path model.pkl
```

---

### 3.5 predict.py

**Mục đích**: Module inference production với tối ưu hóa.

#### Class ModelPredictor

```python
class ModelPredictor
```

**Constructor**:
```python
def __init__(self, model_path: str, enable_cache: bool = True)
```

**Tham số**:
- `model_path`: Đường dẫn file .pkl
- `enable_cache`: Bật cache model (mặc định: True)

**Phương thức chính**:

##### predict_single

```python
def predict_single(self, features: Dict[str, Any], return_proba: bool = True) -> Dict[str, Any]
```

Dự đoán cho 1 instance.

**Tham số**:
- `features`: Dictionary chứa các đặc trưng
- `return_proba`: Trả về xác suất (mặc định: True)

**Trả về**:
```python
{
    'prediction': 0,  # 0 hoặc 1
    'probability': 0.12,  # Xác suất class 1
    'confidence': 0.88  # Độ tin cậy
}
```

**Ví dụ**:
```python
from predict import ModelPredictor

predictor = ModelPredictor('models/best_model.pkl')
result = predictor.predict_single({
    'director_mean_rating': 8.5,
    'director_total_films': 15,
    'cast_mean_rating': 7.8,
    'cast_total_films': 35,
    'writer_mean_rating': 7.5,
    'writer_total_films': 10,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1
})
```

##### predict_batch

```python
def predict_batch(self, features_list: List[Dict], return_proba: bool = True, batch_size: int = None) -> List[Dict]
```

Dự đoán hàng loạt với tối ưu batch.

---

### 3.6 api_server.py

**Mục đích**: REST API server FastAPI.

#### Endpoints Chính

**GET /health** - Kiểm tra sức khỏe
```bash
curl http://localhost:8000/health
```

**POST /predict/single** - Dự đoán 1 phim
```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"director_mean_rating": 8.5, ...}'
```

**POST /predict/batch** - Dự đoán nhiều phim
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"movies": [...], "return_proba": true}'
```

**GET /docs** - Swagger UI documentation

---

## 4. HƯỚNG DẪN TRIỂN KHAI LOCAL

### 4.1 Yêu Cầu Hệ Thống

**Phần cứng**:
- CPU: 4+ cores
- RAM: 8GB tối thiểu, 16GB khuyến nghị
- Ổ đĩa: 5GB trống

**Phần mềm**:
- Python 3.10+
- pip phiên bản mới nhất

### 4.2 Cài Đặt

#### Bước 1: Tạo môi trường ảo

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

#### Bước 2: Cài đặt dependencies

```bash
pip install pandas numpy scikit-learn xgboost joblib
pip install fastapi uvicorn pydantic  # Cho API server
```

Hoặc cài tất cả:
```bash
pip install -r requirements.txt
```

#### Bước 3: Kiểm tra cài đặt

```bash
python -c "import pandas; import sklearn; print('OK')"
```

### 4.3 Chuẩn Bị Dữ Liệu

Đặt các file IMDb vào thư mục `data/`:
- title.basics.tsv
- title.ratings.tsv
- title.crew.tsv
- title.principals.tsv
- name.basics.tsv
- title.akas.tsv

Nguồn: https://datasets.imdbws.com/

### 4.4 Huấn Luyện Mô Hình

#### Cách 1: Huấn luyện tất cả (Khuyến nghị)

```bash
cd src/
python main.py --mode evaluate --data-path ../data/
```

Kết quả sẽ được lưu trong `evaluation_results/best_models/`

#### Cách 2: Huấn luyện từng loại

**Phân loại**:
```bash
python main.py --mode train --model-type classification --data-path ../data/
```

**Regression**:
```bash
python main.py --mode train --model-type regression --data-path ../data/
```

### 4.5 Sử Dụng Mô Hình

#### Sử dụng main.py

**Chế độ tương tác**:
```bash
python main.py --mode interactive --model-path evaluation_results/best_models/best_classification.pkl
```

**Dự đoán hàng loạt**:
```bash
python main.py --mode predict \
  --model-path evaluation_results/best_models/best_classification.pkl \
  --input-file ../test_movies.csv \
  --output-file ../predictions.csv
```

#### Sử dụng predict.py (Production)

```python
from predict import ModelPredictor

predictor = ModelPredictor('evaluation_results/best_models/best_classification.pkl')

# Dự đoán đơn
result = predictor.predict_single({
    'director_mean_rating': 8.5,
    'director_total_films': 15,
    'cast_mean_rating': 7.8,
    'cast_total_films': 35,
    'writer_mean_rating': 7.5,
    'writer_total_films': 10,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1
    # Các genre khác đặt = 0
})
print(result)
```

### 4.6 Chạy API Server

**Development mode**:
```bash
cd src/
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Production mode** (4 workers):
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Truy cập documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Test API**:
```bash
# Health check
curl http://localhost:8000/health

# Dự đoán
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "director_mean_rating": 8.5,
    "director_total_films": 15,
    "cast_mean_rating": 7.8,
    "cast_total_films": 35,
    "writer_mean_rating": 7.5,
    "writer_total_films": 10,
    "runtimeMinutes": 148,
    "startYear": 2023,
    "genre_Action": 1,
    "genre_Drama": 1
  }'
```

---

## 5. VÍ DỤ SỬ DỤNG

### 5.1 Workflow Hoàn Chỉnh

```python
# Bước 1: Xử lý dữ liệu
from preprocess import preprocess_pipeline
df = preprocess_pipeline('../data/')

# Bước 2: Feature engineering
from feature import encode_genres, get_feature_target
df_encoded = encode_genres(df)
X, y = get_feature_target(df_encoded, target_type='is_risky')

# Bước 3: Huấn luyện
from train import train_is_risky_models
models = train_is_risky_models(df)
best_model = models['rf']

# Bước 4: Dự đoán
from predict import ModelPredictor
predictor = ModelPredictor('models/rf_is_risky.pkl')

phim_moi = {
    'director_mean_rating': 8.5,
    'director_total_films': 15,
    'cast_mean_rating': 7.8,
    'cast_total_films': 35,
    'writer_mean_rating': 7.5,
    'writer_total_films': 10,
    'runtimeMinutes': 148,
    'startYear': 2023,
    'genre_Action': 1,
    'genre_Drama': 1
}

ket_qua = predictor.predict_single(phim_moi)
print(f"Dự đoán: {ket_qua}")
```

### 5.2 Xử Lý Hàng Loạt

```python
from predict import predict_from_csv

# Xử lý file CSV
df_results = predict_from_csv(
    model_path='models/best_classification.pkl',
    input_csv='danh_sach_phim.csv',
    output_csv='ket_qua_du_doan.csv'
)

print(df_results[['prediction', 'probability', 'recommendation']])
```

### 5.3 Sử Dụng API

```python
import requests

API_URL = "http://localhost:8000"

# Dự đoán đơn
response = requests.post(
    f"{API_URL}/predict/single",
    json={
        "director_mean_rating": 8.5,
        "director_total_films": 15,
        "cast_mean_rating": 7.8,
        "cast_total_films": 35,
        "writer_mean_rating": 7.5,
        "writer_total_films": 10,
        "runtimeMinutes": 148,
        "startYear": 2023,
        "genre_Action": 1,
        "genre_Drama": 1
    }
)

ket_qua = response.json()
print(f"Dự đoán: {ket_qua['prediction']}")
print(f"Mức độ rủi ro: {ket_qua['risk_level']}")
print(f"Khuyến nghị: {ket_qua['recommendation']}")
```

---

## 6. API REFERENCE

### 6.1 Tổng Quan Endpoints

| Phương thức | Endpoint | Mục đích |
|-------------|----------|----------|
| GET | / | Thông tin API |
| GET | /health | Kiểm tra sức khỏe |
| GET | /docs | Swagger UI |
| GET | /model/info | Metadata model |
| POST | /predict/single | Dự đoán 1 phim |
| POST | /predict/batch | Dự đoán nhiều phim |

### 6.2 Mã Trạng Thái

- 200: Thành công
- 400: Yêu cầu không hợp lệ
- 422: Lỗi validation
- 500: Lỗi server
- 503: Service không khả dụng

### 6.3 Request/Response Format

**Single Prediction Request**:
```json
{
  "director_mean_rating": 8.5,
  "director_total_films": 15,
  "cast_mean_rating": 7.8,
  "cast_total_films": 35,
  "writer_mean_rating": 7.5,
  "writer_total_films": 10,
  "runtimeMinutes": 148,
  "startYear": 2023,
  "genre_Action": 1,
  "genre_Drama": 1
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "confidence": 0.88,
  "risk_level": "LOW",
  "recommendation": "Rủi ro thấp - An toàn để đầu tư"
}
```

---

## PHỤ LỤC A: Cấu Trúc Thư Mục

```
project/
├── data/                  # Dữ liệu IMDb
├── src/                   # Mã nguồn
│   ├── preprocess.py
│   ├── feature.py
│   ├── train.py
│   ├── evaluate_models.py
│   ├── main.py
│   ├── predict.py
│   └── api_server.py
├── models/                # Mô hình đã lưu
├── evaluation_results/    # Kết quả đánh giá
├── Dockerfile
├── requirements.txt
└── TAI_LIEU_KY_THUAT.md
```

---

## PHỤ LỤC B: Hiệu Suất Mô Hình

| Mô Hình | F1-Score | AUC-ROC | R² | MAE |
|---------|----------|---------|-----|-----|
| Logistic Regression | 0.76-0.78 | 0.82-0.84 | - | - |
| Random Forest Classifier | 0.78-0.82 | 0.85-0.87 | - | - |
| Random Forest Regressor | - | - | 0.60-0.65 | 0.9-1.1 |
| XGBoost Regressor | - | - | 0.65-0.70 | 0.85-1.0 |
| SimplerDNN | 0.72-0.75 | 0.78-0.82 | 0.62-0.67 | 0.95-1.05 |
| HybridMovieModel | 0.75-0.78 | 0.80-0.85 | 0.68-0.72 | 0.88-0.98 |

---

## PHỤ LỤC C: Xử Lý Lỗi Thường Gặp

### Lỗi: ModuleNotFoundError

**Giải pháp**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Lỗi: Không tìm thấy model

**Giải pháp**:
```bash
python main.py --mode evaluate --data-path ../data/
```

### Lỗi: API không khởi động được

**Giải pháp**:
```bash
# Kiểm tra port
lsof -i :8000

# Dùng port khác
uvicorn api_server:app --port 8001
```

### Lỗi: Hết bộ nhớ

**Giải pháp**:
```python
# Giảm kích thước dữ liệu
df_sample = df.sample(n=10000, random_state=42)

# Hoặc giảm batch size
train_hybrid_dl_models(batch_size=128)
```

---

## PHỤ LỤC D: Thuật Ngữ

- **is_risky**: Nhãn nhị phân (1 = rủi ro cao, 0 = rủi ro thấp)
- **Multi-output regression**: Mô hình dự đoán nhiều đầu ra cùng lúc
- **Person statistics**: Số liệu thống kê lịch sử của đạo diễn/diễn viên/biên kịch
- **Genre encoding**: Chuyển đổi thể loại thành đặc trưng nhị phân
- **Pipeline**: Object sklearn kết hợp tiền xử lý và mô hình

---

KẾT THÚC TÀI LIỆU KỸ THUẬT
