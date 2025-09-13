# House Price Prediction with Feature Engineering
# Task 8: DATA ANALYTICS INTERNSHIP

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HousePricePredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_names = []
        
    def load_and_explore_data(self, file_path):
        """Load data and perform initial exploration"""
        try:
            # For demonstration, creating a synthetic dataset similar to Kaggle House Prices
            # In real scenario, you would load: df = pd.read_csv(file_path)
            self.df = self.create_synthetic_data()
            print("Dataset created successfully!")
        except:
            print("Creating synthetic dataset for demonstration...")
            self.df = self.create_synthetic_data()
            
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        print(f"\nDataset info:")
        print(self.df.info())
        
        return self.df
    
    def create_synthetic_data(self):
        """Create synthetic house price data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic features similar to house price dataset
        data = {
            'LotArea': np.random.normal(10000, 3000, n_samples),
            'OverallQual': np.random.randint(1, 11, n_samples),
            'OverallCond': np.random.randint(1, 11, n_samples),
            'YearBuilt': np.random.randint(1900, 2022, n_samples),
            'TotalBsmtSF': np.random.normal(1000, 300, n_samples),
            'GrLivArea': np.random.normal(1500, 500, n_samples),
            'BedroomAbvGr': np.random.randint(1, 6, n_samples),
            'BathsFull': np.random.randint(1, 4, n_samples),
            'GarageArea': np.random.normal(500, 200, n_samples),
            'Neighborhood': np.random.choice(['Downtown', 'Suburb', 'Rural', 'Urban'], n_samples),
            'HouseStyle': np.random.choice(['1Story', '2Story', 'Split'], n_samples),
            'Foundation': np.random.choice(['Concrete', 'Wood', 'Stone'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Introduce some missing values
        missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
        df.loc[missing_indices[:50], 'GarageArea'] = np.nan
        df.loc[missing_indices[50:], 'TotalBsmtSF'] = np.nan
        
        # Create target variable with realistic relationship
        df['SalePrice'] = (
            50000 + 
            df['OverallQual'] * 15000 +
            df['GrLivArea'] * 80 +
            df['TotalBsmtSF'].fillna(0) * 30 +
            df['GarageArea'].fillna(0) * 40 +
            np.random.normal(0, 10000, n_samples)
        )
        
        return df
    
    def analyze_missing_values(self):
        """Analyze and visualize missing values"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            print("Missing values by column:")
            for col, count in missing_data.items():
                percentage = (count / len(self.df)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
            
            # Visualize missing values
            plt.figure(figsize=(10, 6))
            missing_data.plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.xlabel('Columns')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found!")
    
    def handle_missing_values(self):
        """Handle missing values for numerical and categorical data"""
        print("\n" + "="*50)
        print("HANDLING MISSING VALUES")
        print("="*50)
        
        # Separate numerical and categorical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from numerical columns
        if 'SalePrice' in numerical_cols:
            numerical_cols.remove('SalePrice')
        
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
        
        # Handle numerical missing values (using median)
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in {col} with median: {median_value:.2f}")
        
        # Handle categorical missing values (using mode)
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_value}")
        
        print("Missing values handled successfully!")
        return self.df
    
    def perform_eda(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Distribution of target variable
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.hist(self.df['SalePrice'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Distribution of Sale Price')
        plt.xlabel('Sale Price')
        plt.ylabel('Frequency')
        
        # Log transformation visualization
        plt.subplot(2, 3, 2)
        plt.hist(np.log1p(self.df['SalePrice']), bins=50, alpha=0.7, color='lightgreen')
        plt.title('Distribution of Log(Sale Price)')
        plt.xlabel('Log(Sale Price)')
        plt.ylabel('Frequency')
        
        # Correlation heatmap for numerical features
        plt.subplot(2, 3, 3)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        # Box plots for categorical features
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) >= 2:
            plt.subplot(2, 3, 4)
            sns.boxplot(data=self.df, x=categorical_cols[0], y='SalePrice')
            plt.title(f'Sale Price by {categorical_cols[0]}')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 3, 5)
            sns.boxplot(data=self.df, x=categorical_cols[1], y='SalePrice')
            plt.title(f'Sale Price by {categorical_cols[1]}')
            plt.xticks(rotation=45)
        
        # Scatter plot for important numerical feature
        plt.subplot(2, 3, 6)
        plt.scatter(self.df['GrLivArea'], self.df['SalePrice'], alpha=0.6)
        plt.title('Sale Price vs Living Area')
        plt.xlabel('Above Ground Living Area')
        plt.ylabel('Sale Price')
        
        plt.tight_layout()
        plt.show()
    
    def check_skewness(self):
        """Check and visualize skewness in numerical features"""
        print("\n" + "="*50)
        print("SKEWNESS ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'SalePrice' in numerical_cols:
            numerical_cols.remove('SalePrice')
        
        skewed_features = []
        for col in numerical_cols:
            skewness = self.df[col].skew()
            if abs(skewness) > 0.75:  # Threshold for high skewness
                skewed_features.append((col, skewness))
                print(f"{col}: {skewness:.3f} (Highly Skewed)")
            else:
                print(f"{col}: {skewness:.3f}")
        
        return skewed_features
    
    def apply_transformations(self, skewed_features):
        """Apply log transformation to highly skewed features"""
        print("\n" + "="*50)
        print("APPLYING TRANSFORMATIONS")
        print("="*50)
        
        transformed_cols = []
        for col, skewness in skewed_features:
            # Apply log1p transformation (log(1+x) to handle zeros)
            original_col = f"{col}_original"
            self.df[original_col] = self.df[col].copy()  # Keep original for comparison
            
            # Ensure non-negative values before log transformation
            if self.df[col].min() < 0:
                self.df[col] = self.df[col] - self.df[col].min()
            
            self.df[col] = np.log1p(self.df[col])
            transformed_cols.append(col)
            
            new_skewness = self.df[col].skew()
            print(f"Transformed {col}: {skewness:.3f} -> {new_skewness:.3f}")
        
        # Also transform target variable if highly skewed
        target_skewness = self.df['SalePrice'].skew()
        if abs(target_skewness) > 0.75:
            print(f"Transforming target variable (skewness: {target_skewness:.3f})")
            self.df['SalePrice_log'] = np.log1p(self.df['SalePrice'])
            self.use_log_target = True
        else:
            self.use_log_target = False
        
        return transformed_cols
    
    def encode_categorical_features(self):
        """Apply encoding techniques for categorical features"""
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*50)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Decide encoding strategy based on cardinality
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            print(f"\n{col}: {unique_values} unique values")
            
            if unique_values <= 5:  # Use One-Hot Encoding for low cardinality
                print(f"Applying One-Hot Encoding to {col}")
                # Create dummy variables
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                
            else:  # Use Label Encoding for high cardinality
                print(f"Applying Label Encoding to {col}")
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                self.df.drop(col, axis=1, inplace=True)
        
        print("Categorical encoding completed!")
    
    def create_new_features(self):
        """Create new engineered features"""
        print("\n" + "="*50)
        print("CREATING NEW FEATURES")
        print("="*50)
        
        # Feature engineering examples
        features_created = []
        
        # Age of house
        if 'YearBuilt' in self.df.columns:
            current_year = 2024
            self.df['HouseAge'] = current_year - self.df['YearBuilt']
            features_created.append('HouseAge')
            print("Created: HouseAge (Current Year - Year Built)")
        
        # Total bathrooms
        if 'BathsFull' in self.df.columns:
            # Assuming half baths exist (for demonstration, create random half baths)
            if 'BathsHalf' not in self.df.columns:
                self.df['BathsHalf'] = np.random.randint(0, 2, len(self.df))
            self.df['TotalBaths'] = self.df['BathsFull'] + 0.5 * self.df['BathsHalf']
            features_created.append('TotalBaths')
            print("Created: TotalBaths (Full Baths + 0.5 * Half Baths)")
        
        # Total area
        area_cols = ['GrLivArea']
        if 'TotalBsmtSF' in self.df.columns:
            area_cols.append('TotalBsmtSF')
        if len(area_cols) > 1:
            self.df['TotalSF'] = self.df[area_cols].sum(axis=1)
            features_created.append('TotalSF')
            print("Created: TotalSF (Ground Living Area + Basement Area)")
        
        # Quality-Condition interaction
        if 'OverallQual' in self.df.columns and 'OverallCond' in self.df.columns:
            self.df['QualCond'] = self.df['OverallQual'] * self.df['OverallCond']
            features_created.append('QualCond')
            print("Created: QualCond (Overall Quality × Overall Condition)")
        
        # Rooms per area ratio
        if 'BedroomAbvGr' in self.df.columns and 'GrLivArea' in self.df.columns:
            self.df['RoomsPerSF'] = self.df['BedroomAbvGr'] / (self.df['GrLivArea'] + 1)  # +1 to avoid division by zero
            features_created.append('RoomsPerSF')
            print("Created: RoomsPerSF (Bedrooms per Square Foot)")
        
        print(f"Total new features created: {len(features_created)}")
        return features_created
    
    def prepare_features(self):
        """Prepare final feature set for modeling"""
        print("\n" + "="*50)
        print("PREPARING FEATURES FOR MODELING")
        print("="*50)
        
        # Remove original columns if we kept them for comparison
        cols_to_drop = [col for col in self.df.columns if col.endswith('_original')]
        if cols_to_drop:
            self.df.drop(cols_to_drop, axis=1, inplace=True)
            print(f"Dropped original columns: {cols_to_drop}")
        
        # Prepare features (X) and target (y)
        if self.use_log_target:
            target_col = 'SalePrice_log'
        else:
            target_col = 'SalePrice'
        
        # Drop target columns from features
        feature_cols = [col for col in self.df.columns if not col.startswith('SalePrice')]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        self.feature_names = X.columns.tolist()
        print(f"Final feature set: {len(self.feature_names)} features")
        print(f"Features: {self.feature_names}")
        
        return X, y
    
    def train_and_evaluate_model(self, X, y):
        """Train model and evaluate performance"""
        print("\n" + "="*50)
        print("TRAINING AND EVALUATING MODEL")
        print("="*50)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"\nModel Performance:")
        print(f"Training RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Feature importance
        self.analyze_feature_importance()
        
        # Plot predictions vs actual
        self.plot_predictions(y_test, test_pred)
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae
        }
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importance from Random Forest
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def plot_predictions(self, y_true, y_pred):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(12, 5))
        
        # Predictions vs Actual scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self, file_path=None):
        """Run the complete feature engineering and modeling pipeline"""
        print("="*60)
        print("HOUSE PRICE PREDICTION - FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Step 1: Load and explore data
        self.load_and_explore_data(file_path)
        
        # Step 2: Analyze missing values
        self.analyze_missing_values()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Perform EDA
        self.perform_eda()
        
        # Step 5: Check skewness and apply transformations
        skewed_features = self.check_skewness()
        transformed_cols = self.apply_transformations(skewed_features)
        
        # Step 6: Encode categorical features
        self.encode_categorical_features()
        
        # Step 7: Create new features
        new_features = self.create_new_features()
        
        # Step 8: Prepare features for modeling
        X, y = self.prepare_features()
        
        # Step 9: Train and evaluate model
        metrics = self.train_and_evaluate_model(X, y)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Number of features used: {len(self.feature_names)}")
        print(f"Model R² Score: {metrics['test_r2']:.4f}")
        
        return metrics

# Usage Example
if __name__ == "__main__":
    # Initialize the predictor
    predictor = HousePricePredictor()
    
    # Run the complete pipeline
    # For Kaggle dataset, use: predictor.run_complete_pipeline('path/to/train.csv')
    results = predictor.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING TECHNIQUES DEMONSTRATED:")
    print("="*60)
    print("✓ Missing value analysis and handling")
    print("✓ Exploratory Data Analysis (EDA)")
    print("✓ Skewness detection and log transformation")
    print("✓ One-Hot Encoding for low cardinality categorical features")
    print("✓ Label Encoding for high cardinality categorical features")
    print("✓ Feature creation and engineering")
    print("✓ Feature scaling and standardization")
    print("✓ Model training with Random Forest")
    print("✓ Feature importance analysis")
    print("✓ Model evaluation and visualization")