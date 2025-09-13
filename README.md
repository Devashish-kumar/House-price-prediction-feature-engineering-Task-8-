# House Price Prediction with Feature Engineering

##  Task Overview

**Task 8: Improving House Price Predictions through Feature Engineering**

This project demonstrates comprehensive feature engineering techniques to enhance the predictive performance of a machine learning model using a housing dataset. The implementation covers all major preprocessing steps from handling missing values to creating new features and model evaluation.

##  Objective

To apply effective feature engineering techniques including:
- Handling missing values
- Encoding categorical variables
- Data transformations
- Feature creation
- Model training and evaluation

##  Tools & Technologies Used

- **Python 3.7+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

##  Dataset

The project uses the **House Prices - Advanced Regression Dataset** concept from Kaggle. For demonstration purposes, a synthetic dataset is generated that mimics the structure and characteristics of the original dataset.

### Dataset Features:
- **Numerical Features**: LotArea, OverallQual, OverallCond, YearBuilt, TotalBsmtSF, GrLivArea, BedroomAbvGr, BathsFull, GarageArea
- **Categorical Features**: Neighborhood, HouseStyle, Foundation
- **Target Variable**: SalePrice

##  Feature Engineering Techniques Implemented

### 1. Missing Value Analysis & Handling
- **Detection**: Using `df.isnull().sum()` to identify missing values
- **Numerical Data**: Median imputation for numerical features
- **Categorical Data**: Mode imputation for categorical features
- **Visualization**: Bar plots showing missing value patterns

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of target variable
- Correlation matrix heatmaps
- Box plots for categorical features
- Scatter plots for numerical relationships

### 3. Data Transformations
- **Skewness Detection**: Identifying highly skewed features (threshold: |skewness| > 0.75)
- **Log Transformation**: Using `np.log1p()` for skewed data
- **Target Transformation**: Log transformation of target variable if needed

### 4. Categorical Encoding
- **One-Hot Encoding**: `pd.get_dummies()` for low cardinality features (≤5 unique values)
- **Label Encoding**: For high cardinality categorical features (>5 unique values)
- **Smart Selection**: Automatic encoding strategy based on feature cardinality

### 5. Feature Creation
- **House Age**: Current Year - Year Built
- **Total Bathrooms**: Full Baths + 0.5 × Half Baths
- **Total Square Footage**: Ground Living Area + Basement Area
- **Quality-Condition Interaction**: Overall Quality × Overall Condition
- **Rooms per Square Foot**: Bedrooms per unit area

### 6. Feature Scaling
- **StandardScaler**: Standardization of all numerical features
- **Consistent Scaling**: Applied to both training and test sets

##  Machine Learning Model

### Algorithm: Random Forest Regressor
- **Parameters**: 100 estimators, random_state=42
- **Advantages**: 
  - Handles both numerical and categorical features
  - Provides feature importance scores
  - Robust to outliers
  - Good baseline performance

### Model Evaluation Metrics
- **RMSE** (Root Mean Square Error)
- **R² Score** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)
- **Feature Importance Analysis**

##  Project Structure

```
house-price-prediction/
│
├── house_price_prediction.py    # Main Python script
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── data/                        # Dataset directory
│   └── (your dataset files)
├── results/                     # Output files and plots
│   ├── feature_importance.png
│   ├── predictions_plot.png
│   └── eda_plots.png
└── .gitignore                   # Git ignore file
```

##  How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Using Synthetic Data (Default)
```python
from house_price_prediction import HousePricePredictor

# Initialize the predictor
predictor = HousePricePredictor()

# Run complete pipeline with synthetic data
results = predictor.run_complete_pipeline()
```

#### Option 2: Using Your Own Dataset
```python
# For Kaggle House Prices dataset
predictor = HousePricePredictor()
results = predictor.run_complete_pipeline('path/to/your/train.csv')
```

### Step-by-Step Execution
```python
# Run individual steps
predictor.load_and_explore_data()
predictor.analyze_missing_values()
predictor.handle_missing_values()
predictor.perform_eda()
predictor.encode_categorical_features()
predictor.create_new_features()
X, y = predictor.prepare_features()
metrics = predictor.train_and_evaluate_model(X, y)
```

##  Results & Performance

The pipeline generates comprehensive analysis including:

### Model Performance Metrics
- Training and test RMSE scores
- R² scores for model accuracy
- Mean Absolute Error (MAE)
- Feature importance rankings

### Visualizations Generated
1. **Missing Values Bar Chart**
2. **EDA Dashboard** (6 subplot visualization)
   - Sale price distribution
   - Log-transformed price distribution
   - Correlation matrix heatmap
   - Categorical feature box plots
   - Scatter plots for key relationships
3. **Feature Importance Plot**
4. **Model Performance Plots**
   - Predictions vs Actual values
   - Residual analysis

### Sample Output
```
Model Performance:
Training RMSE: 15420.50
Test RMSE: 18750.25
Training R²: 0.9245
Test R²: 0.8876
Test MAE: 13250.75

Top 5 Most Important Features:
1. OverallQual: 0.3456
2. GrLivArea: 0.2134
3. TotalSF: 0.1876
4. QualCond: 0.1234
5. HouseAge: 0.0987
```

##  Key Learning Outcomes

### Technical Skills Developed
-  **Data Preprocessing**: Comprehensive missing value handling strategies
-  **Feature Engineering**: Creating meaningful features from existing data
-  **Encoding Techniques**: When and how to use different encoding methods
-  **Data Transformation**: Handling skewed data with mathematical transformations
-  **Model Evaluation**: Understanding multiple evaluation metrics
-  **Feature Selection**: Using Random Forest feature importance

### Best Practices Implemented
- Modular, object-oriented code structure
- Comprehensive error handling
- Detailed documentation and comments
- Visualization for better understanding
- Scalable pipeline design

##  Interview Questions & Answers

### Q1: What is feature engineering, and why is it important?
**Answer**: Feature engineering is the process of selecting, transforming, and creating features from raw data to improve machine learning model performance. It's important because:
- Models perform better with well-engineered features
- Helps capture domain knowledge
- Can reveal hidden patterns in data
- Often more impactful than choosing different algorithms

### Q2: When would you use label encoding vs one-hot encoding?
**Answer**: 
- **Label Encoding**: Use for ordinal categorical data (inherent order) or high cardinality features
- **One-Hot Encoding**: Use for nominal categorical data (no inherent order) with low cardinality
- **Rule of Thumb**: One-hot for ≤5 categories, label encoding for >5 categories

### Q3: How do you handle missing values for categorical and numerical data?
**Answer**:
- **Numerical Data**: Mean/Median imputation, forward/backward fill, or domain-specific values
- **Categorical Data**: Mode imputation, "Unknown" category, or most frequent category
- **Advanced**: Multiple imputation, predictive imputation, or domain expertise

### Q4: What are some common transformations to reduce skewness in data?
**Answer**:
- **Log Transformation**: `log(x+1)` for right-skewed data
- **Square Root**: `sqrt(x)` for moderate skewness
- **Box-Cox Transformation**: Optimal lambda parameter transformation
- **Yeo-Johnson**: Works with negative values

### Q5: How do you determine which features to keep or remove?
**Answer**:
- **Statistical Methods**: Correlation analysis, mutual information
- **Model-Based**: Feature importance from tree models, recursive feature elimination
- **Domain Knowledge**: Subject matter expertise
- **Performance-Based**: Cross-validation with different feature sets

## 🔧 Customization Options

### Adding New Features
```python
def create_custom_feature(self):
    # Add your custom feature engineering logic
    self.df['NewFeature'] = self.df['Feature1'] / self.df['Feature2']
    return ['NewFeature']
```

### Changing Model
```python
from sklearn.ensemble import GradientBoostingRegressor
self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
```

### Different Encoding Strategies
```python
# Target encoding for high cardinality
from category_encoders import TargetEncoder
encoder = TargetEncoder()
```

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is created for educational purposes as part of the Data Analytics Internship program.

##  Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

##  Acknowledgments

- Kaggle for the House Prices dataset concept
- Data Analytics Internship Program
- scikit-learn community for excellent documentation
- Open source Python data science community
