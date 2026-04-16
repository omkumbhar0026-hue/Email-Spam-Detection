# 📧 Email Spam Detection using Machine Learning

## 📌 Problem Statement

This project aims to classify emails as **spam or not spam (ham)** using machine learning and Natural Language Processing (NLP) techniques. Spam detection is a crucial task in cybersecurity to prevent phishing attacks and unwanted messages.

---

## 🧠 Techniques Used

### 🔹 Text Preprocessing

* Lowercasing
* Removal of stopwords
* Tokenization and cleaning

### 🔹 Feature Extraction

* **TF-IDF Vectorization** to convert text data into numerical form

### 🔹 Machine Learning Models

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

---

## 📊 Results

The models were evaluated using **Accuracy, ROC-AUC, and Cross-Validation scores**.

| Model               | Accuracy   | ROC-AUC    | CV Score   |
| ------------------- | ---------- | ---------- | ---------- |
| Logistic Regression | 87.40%     | 0.9524     | 0.8898     |
| Random Forest       | 88.80%     | 0.9604     | 0.9001     |
| Gradient Boosting   | **89.35%** | **0.9672** | **0.9065** |
| SVM                 | 87.55%     | 0.9492     | 0.8914     |
| KNN                 | 84.55%     | 0.9145     | 0.8592     |

### 🏆 Best Model

**Gradient Boosting** performed the best due to its ability to combine multiple weak learners and reduce both bias and variance.

---

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── data/
│   ├── spam_email_dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── starter_code.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Usage

1. Clone the repository:

```
git clone https://github.com/your-username/Email-Spam-Detection.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the notebook or starter code:

```
python starter_code.py
```

---

## 📂 Dataset

The dataset contains labeled email messages categorized as:

* **Spam (1)**
* **Ham (0)**

---

## 🚀 Key Highlights

* Applied **NLP techniques** for real-world text classification
* Compared multiple ML models
* Used **TF-IDF vectorization** for feature extraction
* Evaluated using multiple performance metrics

---

## 🔮 Future Scope

* Improve performance using **Deep Learning (LSTM / Transformers)**
* Deploy as a **web application**
* Extend to **phishing and scam detection systems**

---

## 👨‍💻 Author

Om Kumbhar
