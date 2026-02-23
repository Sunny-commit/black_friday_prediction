# 🛍️ Black Friday Sales Prediction - Time-Based Forecasting

A **machine learning project predicting customer purchase amounts** during Black Friday using historical transaction data and customer behavior patterns.

## 🎯 Overview

This project provides:
- ✅ Time-based data splitting
- ✅ Customer segmentation
- ✅ Product category analysis
- ✅ Regression models for sales prediction
- ✅ Feature importance analysis
- ✅ Seasonal pattern detection

## 📊 Dataset Structure

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BlackFridayDataLoader:
    """Load and explore Black Friday data"""
    
    def __init__(self, filepath='BlackFriday.csv'):
        self.df = pd.read_csv(filepath)
    
    def dataset_overview(self):
        """Dataset summary"""
        print(f"Shape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
    
    def customer_analysis(self):
        """Analyze customer demographics"""
        print(f"Total customers: {self.df['User_ID'].nunique()}")
        print(f"Gender distribution:\n{self.df['Gender'].value_counts()}")
        print(f"Age distribution:\n{self.df['Age'].value_counts().sort_index()}")
        print(f"City category:\n{self.df['City_Category'].value_counts()}")
    
    def product_analysis(self):
        """Analyze products"""
        print(f"Total products: {self.df['Product_ID'].nunique()}")
        print(f"Product categories:\n{self.df['Product_Category_1'].value_counts()}")
        print(f"Average price by category:")
        print(self.df.groupby('Product_Category_1')['Purchase'].mean().sort_values(ascending=False))
```

## 🔧 Feature Engineering

```python
class BlackFridayFeatureEngineer:
    """Create predictive features"""
    
    def __init__(self):
        self.customer_stats = None
        self.product_stats = None
    
    def customer_features(self, df):
        """Aggregate customer-level features"""
        customer_agg = df.groupby('User_ID').agg({
            'Purchase': ['sum', 'mean', 'count', 'std'],
            'Product_ID': 'nunique',
            'Product_Category_1': 'nunique'
        }).reset_index()
        
        customer_agg.columns = ['User_ID', 'total_purchase', 'avg_purchase', 
                               'purchase_count', 'purchase_std',
                               'unique_products', 'unique_categories']
        
        customer_agg['purchase_std'].fillna(0, inplace=True)
        
        self.customer_stats = customer_agg
        return customer_agg
    
    def product_features(self, df):
        """Aggregate product-level features"""
        product_agg = df.groupby('Product_ID').agg({
            'Purchase': ['mean', 'count', 'std'],
            'User_ID': 'nunique'
        }).reset_index()
        
        product_agg.columns = ['Product_ID', 'avg_price', 'popularity',
                              'price_std', 'unique_customers']
        
        product_agg['price_std'].fillna(0, inplace=True)
        
        self.product_stats = product_agg
        return product_agg
    
    def demographic_features(self, df):
        """Create demographic features"""
        df_copy = df.copy()
        
        # Age encoding
        age_mapping = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, 
                      '46-50': 4, '51-55': 5, '55+': 6}
        df_copy['Age_encoded'] = df_copy['Age'].map(age_mapping)
        
        # Gender encoding
        df_copy['Gender_encoded'] = (df_copy['Gender'] == 'M').astype(int)
        
        # City encoding
        city_mapping = {'A': 0, 'B': 1, 'C': 2}
        df_copy['City_encoded'] = df_copy['City_Category'].map(city_mapping)
        
        return df_copy
```

## 🤖 Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

class BlackFridayRegressor:
    """Regression models for sales prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._build_models()
    
    def _build_models(self):
        """Initialize models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=10.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVR': SVR(kernel='rbf', C=1000)
        }
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        trained_models = {}
        
        # Scale features for distance-based models
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            if name == 'SVR':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            trained_models[name] = model
        
        return trained_models
    
    def predict_batch(self, X_test, trained_models):
        """Make predictions"""
        predictions = {}
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in trained_models.items():
            if name == 'SVR':
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X_test)
            
            predictions[name] = pred
        
        return predictions
```

## 📊 Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BlackFridayEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """Calculate metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2
        }
    
    @staticmethod
    def compare_all_models(y_true, predictions_dict):
        """Compare models"""
        results = {}
        
        for model_name, y_pred in predictions_dict.items():
            results[model_name] = BlackFridayEvaluator.regression_metrics(y_true, y_pred)
        
        results_df = pd.DataFrame(results).T.sort_values('R²', ascending=False)
        print("\nModel Comparison:")
        print(results_df)
        
        return results_df
    
    @staticmethod
    def feature_importance(model, feature_names):
        """Extract feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(importance_df.head(10))
            
            return importance_df
```

## 💡 Interview Talking Points

**Q: Handle non-numeric data?**
```
Answer:
- One-hot encoding: Categorical to binary
- Label encoding: Ordinal data
- Target encoding: Category → mean target value
```

**Q: Prevent overfitting?**
```
Answer:
- Train-validation-test split
- Cross-validation
- Regularization (Ridge/Lasso)
- Early stopping (Gradient Boosting)
```

## 🌟 Portfolio Value

✅ Customer behavior analysis
✅ Feature engineering
✅ Sales forecasting
✅ Model comparison
✅ Feature importance
✅ E-commerce domain knowledge

---

**Technologies**: Scikit-learn, Pandas, NumPy, Matplotlib

