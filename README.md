# Collaborative Filtering Recommender System

This project demonstrates how to build recommendation systems using collaborative filtering. It includes both classic matrix factorization techniques and deep learning approaches to predict user preferences based on historical interactions.

## 📁 Project Structure
```
project_root/
├── README.md                         # Project documentation
├── requirements.txt                  # Dependencies
├── data/
│   └── sample_ratings.csv           # Sample dataset (userId, itemId, rating)
├── notebooks/
│   └── Collaborative_Filtering_Exploration.ipynb  # Data exploration & model insights
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # CSV data loading & validation
│   ├── matrix_factorization.py      # SVD-style collaborative filtering
│   └── deep_learning_model.py       # Neural CF with embedding layers
└── main.py                           # Entry point to run training & evaluation
```

## 🚀 Features
- Matrix factorization using Singular Value Decomposition (SVD)
- Deep learning model using user/item embeddings
- Modular code with reusable components
- Jupyter notebook for visualization and experimentation

## 🧪 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/DineshS-Dev/collaborative-filtering-recommender.git
cd collaborative-filtering-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```

## 📊 Sample Dataset
The sample ratings file (`data/sample_ratings.csv`) should include the following columns:
```
userId,itemId,rating
1,101,4.0
1,102,5.0
2,101,3.0
...
```

## 📌 TODO
- Add model evaluation metrics (RMSE, MAE)
- Improve deep learning model with dropout and batch normalization
- Add hyperparameter tuning

---

Feel free to fork, use, and improve it. PRs welcome! 🚀

---

