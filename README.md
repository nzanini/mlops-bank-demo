# MLOps Bank Demo

This project is an example of an MLOps workflow using **Python**, **DVC**, and **MLflow** to train a classification model on a sample banking dataset.

---

## 🚀 What does this project do?

✅ Downloads example banking data  
✅ Preprocesses the data (cleaning and train/test split)  
✅ Trains a Random Forest model  
✅ Logs metrics and parameters with MLflow  
✅ Versions data and the pipeline with DVC  
✅ Allows you to reproduce everything with a single command

---

## 📂 Repository Structure

```
.
├── data/             # Raw and processed data (versioned with DVC)
├── models/           # Trained models
├── mlruns/           # MLflow experiment logs
├── src/              # Python scripts
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   └── train.py
├── dvc.yaml          # DVC pipeline definition
├── dvc.lock          # Pipeline lock file
├── requirements.txt  # Python dependencies
├── .gitignore
└── .dvcignore
```

---

## ⚙️ How to Use

1️⃣ **Clone the repository**
```bash
git clone https://github.com/nzanini/mlops-bank-demo.git
cd mlops-bank-demo
```

2️⃣ **Create and activate a virtual environment**
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Reproduce the pipeline**
```bash
dvc repro
```

5️⃣ **View experiments in MLflow**
```bash
mlflow ui
```
Open in your browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Technologies Used

- **Python**
- **scikit-learn**
- **pandas**
- **DVC**
- **MLflow**
- **Git**

---

## 📈 Expected Results

After running the pipeline, you will have:

✅ A trained model saved in `models/`  
✅ Preprocessed data in `data/processed/`  
✅ Metrics and parameters logged in MLflow  

---

## ✨ Next Steps

- Configure a DVC remote storage (e.g., S3, Google Drive)
- Deploy the model using FastAPI
- Create a GitHub Actions workflow for automated CI/CD

---

## 📄 License

This project is provided for educational purposes. Feel free to use and adapt it.
