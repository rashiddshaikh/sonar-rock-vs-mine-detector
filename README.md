# 🌊 Sonar Rock vs Mine Detector

A machine learning project that classifies sonar signals to detect whether the object is a **rock** or a **mine** using the **Sonar dataset** from UCI Machine Learning Repository.

> This project demonstrates the end-to-end ML pipeline — including data preprocessing, model training, evaluation, and deployment with a web interface.

---

## 🚀 Features

- 🔍 Binary classification: Rock (R) vs Mine (M)
- 📊 Visualizations of model performance
- 🧠 Trained using Logistic Regression / Random Forest
- 🌐 Deployed using Streamlit for web interaction
- 📁 Uses Sonar dataset from UCI ML Repo
- ✅ Accuracy, precision, and confusion matrix reports

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit (for frontend)
- Joblib / Pickle (for model saving)

---

## 📁 Folder Structure

```bash
sonar-rock-vs-mine-detector/
├── sonar_model.pkl           # Trained ML model
├── sonar.csv                 # Dataset file
├── sonar_predictor.py        # Streamlit web app
├── sonar_train.ipynb         # Model training notebook
├── requirements.txt
└── README.md
```

⚙️ Installation & Usage
🔃 Clone the Repository
```bash
git clone https://github.com/rashiddshaikh/sonar-rock-vs-mine-detector.git
cd sonar-rock-vs-mine-detector
```
🐍 Create Virtual Environment (Optional but Recommended)
```bash
python -m venv env
env\Scripts\activate   # Windows
# or
source env/bin/activate  # macOS/Linux
```
📦 Install Dependencies
```bash
pip install -r requirements.txt
```
▶️ Run the Web App (Streamlit)
```bash
streamlit run sonar_predictor.py
```
Visit: http://localhost:8501


🧠 Model Training
To retrain the model:

Open sonar_train.ipynb in Jupyter or Colab.

Load and preprocess the dataset.

Train using Logistic Regression / RandomForest.

Save the model:

```python
import joblib
joblib.dump(model, 'sonar_model.pkl')
```

