:

🛒 Black Friday Sales Prediction
This project analyzes and models consumer purchasing behavior using the Black Friday Sales Dataset provided by a retail company. The goal is to predict the Purchase amount made by customers based on various demographic and transactional features.

📄 Dataset Overview
The dataset includes information such as:

User demographics (Gender, Age, City_Category, Stay_In_Current_City_Years)

Product categories (Product_Category_1, 2, 3)

Purchase amount

🚀 Key Steps in the Pipeline
📌 Data Cleaning & Preprocessing
Handled missing values using mode imputation

Converted categorical variables using Label Encoding

Removed irrelevant features (User_ID, Product_ID)

Verified dataset integrity and structure

📊 Exploratory Data Analysis
Visualized distributions of numerical features using histograms and distplots

Analyzed categorical feature counts using count plots

Reviewed correlations and unique value counts for better understanding of data spread

🔧 Model Development
Train/Test Split: 80/20 ratio

Model Used: Random Forest Regressor

Evaluation Metrics:

R² Score

Mean Squared Error (MSE)

📈 Feature Importance
The model highlights which features had the greatest impact on purchase predictions via a bar chart.

🛠️ Libraries Used
Python (pandas, numpy)

Visualization: seaborn, matplotlib

Machine Learning: scikit-learn

✅ Results
The trained model provides a good baseline for predicting customer purchase behavior and can be further enhanced using:

Feature engineering

Model tuning (GridSearchCV, etc.)

Ensemble or deep learning models

