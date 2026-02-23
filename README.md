# 🖤 Black Friday Prediction - Sales Forecasting

An **end-to-end machine learning system** for predicting Black Friday purchase amounts using customer and product features.

## 🎯 Overview

This project covers:
- ✅ Customer segmentation
- ✅ Feature engineering
- ✅ Regression models
- ✅ Price elasticity
- ✅ Sales forecasting
- ✅ Business insights
- ✅ Campaign optimization

## 👥 Customer Data Analysis

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CustomerAnalyzer:
    """Analyze customer behavior"""
    
    def __init__(self):
        self.data = None
    
    def load_customer_data(self, filepath):
        """Load Black Friday data"""
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def segment_customers(self):
        """Customer segmentation"""
        df = self.data.copy()
        
        # Purchase frequency
        purchase_freq = df.groupby('customer_id').size()
        df['purchase_frequency'] = df['customer_id'].map(purchase_freq)
        
        # Average purchase value
        avg_purchase = df.groupby('customer_id')['purchase_amount'].mean()
        df['avg_purchase_value'] = df['customer_id'].map(avg_purchase)
        
        # Customer lifetime value proxy
        df['customer_lifetime_value'] = df.groupby('customer_id')['purchase_amount'].transform('sum')
        
        # Segment by value
        df['customer_segment'] = pd.qcut(
            df['customer_lifetime_value'],
            q=4,
            labels=['low_value', 'medium_value', 'high_value', 'premium']
        )
        
        return df
    
    def analyze_demographics(self):
        """Analyze demographic patterns"""
        df = self.data.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 18, 25, 35, 50, 100],
                                labels=['teen', 'young', 'adult', 'middle', 'senior'])
        
        # Gender analysis
        gender_stats = df.groupby('gender').agg({
            'purchase_amount': ['mean', 'sum', 'count']
        })
        
        # City tier analysis
        city_stats = df.groupby('city_tier').agg({
            'purchase_amount': ['mean', 'std', 'count']
        })
        
        return {
            'gender_stats': gender_stats,
            'city_stats': city_stats,
            'age_groups': df['age_group'].value_counts()
        }
    
    def analyze_product_preferences(self):
        """Product preference patterns"""
        df = self.data.copy()
        
        # Category preferences by gender/age
        category_gender = df.groupby(['gender', 'product_category']).agg({
            'purchase_amount': ['mean', 'count']
        })
        
        # Product affinity
        high_value_categories = df[df['purchase_amount'] > df['purchase_amount'].quantile(0.75)][
            'product_category'
        ].value_counts()
        
        return {
            'category_gender': category_gender,
            'high_value_categories': high_value_categories
        }
```

## 🛍️ Feature Engineering

```python
class BlackFridayFeatureEngineer:
    """Engineer features for prediction"""
    
    @staticmethod
    def create_purchase_features(data):
        """Purchase behavior features"""
        df = data.copy()
        
        # Purchase amount statistics
        df['log_purchase_amount'] = np.log1p(df['purchase_amount'])
        
        # Spending per item
        if 'quantity' in df.columns:
            df['price_per_item'] = df['purchase_amount'] / (df['quantity'] + 1)
        
        # Is high-ticket purchase
        df['is_high_value'] = (df['purchase_amount'] > df['purchase_amount'].quantile(0.75)).astype(int)
        
        return df
    
    @staticmethod
    def create_temporal_features(data):
        """Time-based features"""
        df = data.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    @staticmethod
    def create_customer_features(data):
        """Aggregate customer features"""
        df = data.copy()
        
        # Customer RFM
        customer_data = df.groupby('customer_id').agg({
            'purchase_amount': ['sum', 'mean', 'count', 'std'],
            'product_category': 'nunique'
        }).fillna(0)
        
        customer_data.columns = ['total_spending', 'avg_purchase', 'purchase_count', 
                                 'spending_std', 'category_diversity']
        
        df = df.merge(customer_data, left_on='customer_id', right_index=True)
        
        return df
    
    @staticmethod
    def encode_features(data):
        """Encode categorical variables"""
        df = data.copy()
        
        # Gender encoding
        df['is_male'] = (df['gender'] == 'M').astype(int)
        
        # City tier encoding
        city_order = {'Tier_1': 3, 'Tier_2': 2, 'Tier_3': 1}
        df['city_tier_numeric'] = df['city_tier'].map(city_order)
        
        # One-hot encode product category if needed
        df = pd.get_dummies(df, columns=['product_category'], prefix='cat')
        
        return df
```

## 🧠 Prediction Models

```python
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

class SalesPredictor:
    """Predict purchase amounts"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
    
    def ridge_regression(self, X_train, y_train):
        """Ridge regression baseline"""
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        self.models['ridge'] = ridge
        
        return ridge
    
    def gradient_boosting(self, X_train, y_train):
        """Gradient Boosting for sales"""
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        self.models['gb'] = gb
        
        return gb
    
    def random_forest(self, X_train, y_train):
        """Random Forest for sales"""
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        self.models['rf'] = rf
        
        return rf
    
    def predict_customer_purchase(self, customer_features):
        """Predict purchase for customer"""
        if self.best_model is None:
            raise ValueError("Model not trained")
        
        prediction = self.best_model.predict([customer_features])[0]
        
        # Ensure positive prediction
        return max(0, prediction)
    
    def segment_customers_by_spending(self, X, predictions):
        """Segment by predicted spending"""
        df = pd.DataFrame({'predicted_amount': predictions})
        
        df['spending_segment'] = pd.qcut(
            df['predicted_amount'],
            q=4,
            labels=['low', 'medium', 'high', 'premium'],
            duplicates='drop'
        )
        
        return df
```

## 📊 Business Insights

```python
class CampaignOptimizer:
    """Optimize marketing campaigns"""
    
    @staticmethod
    def customer_lifetime_value(transactions_df):
        """Calculate CLV for each customer"""
        clv = transactions_df.groupby('customer_id')['purchase_amount'].sum()
        
        return clv
    
    @staticmethod
    def optimal_discount_strategy(price, elasticity=-1.5):
        """Calculate optimal discount"""
        # Price elasticity formula
        # % change in quantity = elasticity * % change in price
        
        optimal_discount = -1 / elasticity
        
        return optimal_discount * 100
    
    @staticmethod
    def recommendation_priority(df, predictions):
        """Rank customers for targeted offers"""
        df['predicted_spending'] = predictions
        df['priority_score'] = (
            (df['predicted_spending'] / df['predicted_spending'].max()) * 0.5 +
            (df['customer_lifetime_value'] / df['customer_lifetime_value'].max()) * 0.3 +
            (df['purchase_frequency'] / df['purchase_frequency'].max()) * 0.2
        )
        
        return df.sort_values('priority_score', ascending=False)
```

## 💡 Interview Talking Points

**Q: Price elasticity implications?**
```
Answer:
- Elasticity = % change volume / % change price
- Elastic: small price change → big volume change
- Inelastic: price change ~ volume stable
- Revenue = price × volume
- Optimize for revenue not just volume
```

**Q: Black Friday prediction challenges?**
```
Answer:
- One event yearly (limited training data)
- External factors (economy, trends)
- Customer behavior shift during sales
- Stock limitations (inventory)
- Competitor actions unpredictable
```

## 🌟 Portfolio Value

✅ Customer segmentation
✅ RFM analysis
✅ Feature engineering
✅ Regression modeling
✅ Business metrics
✅ Campaign optimization
✅ Actionable insights

---

**Technologies**: Scikit-learn, Pandas, NumPy

