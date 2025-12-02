# Dependency Audit Report

**Date**: 2025-11-21  
**Project**: Movie Risk Prediction ML System

## 📋 Complete Import Analysis

### Core Python Files Audited
- ✅ `src/preprocess.py`
- ✅ `src/feature.py`
- ✅ `src/train.py`
- ✅ `src/evaluate_models.py`
- ✅ `src/main.py`
- ✅ `src/predict.py`
- ✅ `src/api_server.py`
- ✅ `utilities/helper.py`

---

## 📦 Required Dependencies

### 1. **Core Data Science** (REQUIRED)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `pandas` | >=2.0.3 | All files | DataFrame operations |
| `numpy` | >=1.24.3 | All files | Numerical computing |
| `matplotlib` | >=3.7.0 | evaluate_models.py | Plotting |
| `seaborn` | >=0.12.0 | evaluate_models.py | Statistical visualization |

### 2. **Machine Learning** (REQUIRED)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `scikit-learn` | >=1.3.0 | train.py, evaluate_models.py | ML algorithms, metrics |
| `xgboost` | >=1.7.6 | train.py, evaluate_models.py | XGBoost regressor |
| `joblib` | >=1.3.1 | train.py, main.py, predict.py | Model serialization |

### 3. **Deep Learning** (OPTIONAL)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `torch` | >=2.0.1 | train.py | Neural network framework |
| `pytorch-lightning` | >=2.0.6 | train.py | DL training framework |

**Note**: Only needed if using `SimplerDNN` or `HybridMovieModel`

### 4. **API Server** (REQUIRED for production)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `fastapi` | >=0.101.0 | api_server.py | REST API framework |
| `uvicorn` | >=0.23.2 | api_server.py | ASGI server |
| `pydantic` | >=2.1.1 | api_server.py | Data validation |

### 5. **Utilities** (REQUIRED)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `python-dotenv` | >=1.0.0 | N/A (future use) | Environment variables |
| `tqdm` | >=4.65.0 | preprocess.py, evaluate_models.py | Progress bars |

### 6. **Testing** (OPTIONAL)
| Package | Version | Used In | Purpose |
|---------|---------|---------|---------|
| `pytest` | >=7.4.0 | N/A | Unit testing |
| `requests` | >=2.31.0 | N/A | HTTP testing |

---

## 🔍 Standard Library Imports (No installation needed)

The following are Python standard library modules:
- `os`, `sys` - File system, system operations
- `json` - JSON serialization
- `argparse` - CLI argument parsing
- `warnings` - Warning control
- `typing` - Type hints
- `datetime` - Date/time handling
- `logging` - Logging framework
- `math` - Mathematical functions
- `pathlib` - Path operations

---

## ✅ Verification Results

### Missing Dependencies: **NONE**
All imports in the codebase are covered by `requirements.txt`

### Recommended Installation Commands

**For basic ML training (without Deep Learning):**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib tqdm
```

**For complete system (including API server):**
```bash
pip install -r requirements.txt
```

**For development (including DL and testing):**
```bash
# Uncomment torch, pytorch-lightning, pytest, requests in requirements.txt
pip install -r requirements.txt
```

---

## 📌 Notes

1. **Deep Learning packages** are commented out by default to reduce installation size
2. **Testing packages** are optional for development
3. All core ML functionality works without DL packages
4. API server requires `fastapi`, `uvicorn`, `pydantic`
5. Progress bars require `tqdm` (added for better UX)

---

## 🎯 Status: ✅ COMPLETE

All dependencies verified and documented. `requirements.txt` is production-ready.
