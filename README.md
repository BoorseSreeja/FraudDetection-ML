# Fraud Detection Using Machine Learning

## Author
**B. Sreeja**  
Location: Bengaluru, Karnataka, India  
Email: sreejaboorse@gmail.com  
LinkedIn: [linkedin.com/in/sreeja-boorse-b1641625b](https://www.linkedin.com/in/sreeja-boorse-b1641625b)  

---

## Project Overview
This project aims to **detect fraudulent online financial transactions** using **Machine Learning**. A **Random Forest Classifier** is used to predict fraud based on transaction details. The workflow includes:

- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Model training  
- Model evaluation  
- Visualizing results  

This project helps financial institutions identify suspicious transactions and reduce losses due to fraud.

---

## Repository Files
| File | Description |
|------|-------------|
| [`Report.pdf`](Report.pdf) | Detailed project report including analysis, methodology, and results |
| [`Dataset.png`](Dataset.png) | Sample visualization of the dataset or EDA |
| [`Code.py`](Code.py) | Python script containing data preprocessing, EDA, model training, and evaluation |

---

## Dataset
The dataset contains transaction information including:

- `nameOrig`, `nameDest` – Customer IDs (removed during preprocessing)  
- `type` – Transaction type (encoded numerically)  
- `amount` – Transaction amount  
- `isFraud` – Target variable (0 = Not Fraud, 1 = Fraud)  

> Note: The dataset visualization is included as `Dataset.png`. The actual CSV data should be uploaded to run the code.

---

## How to Run
1. Clone the repository:
```bash
git clone <your-repo-link>
