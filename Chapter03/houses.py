import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

# Set up logging to document preprocessing decisions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_examine_data(train_path='train.csv', test_path='test.csv'):
    """Load Kaggle House Prices data and perform initial examination."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Test set shape: {test_df.shape}")
    
    # Check for demographic variables that might introduce bias
    demographic_cols = ['Neighborhood', 'MSSubClass']
    for col in demographic_cols:
        logger.info(f"\n{col} distribution:\n{train_df[col].value_counts(normalize=True)[:5]}")
    
    return train_df, test_df

def identify_features(df):
    """Categorize features based on their data types and content."""
    # Start with all columns except Id and SalePrice
    all_features = [col for col in df.columns if col not in ['Id', 'SalePrice']]
    
    # Initialize feature lists
    numeric_features = []
    categorical_features = []
    
    # Examine each feature
    for feature in all_features:
        # Check if the column contains any string values
        if df[feature].dtype == object or df[feature].dtype == 'category':
            categorical_features.append(feature)
        else:
            # For numeric columns, check if they're actually categorical
            if feature in ['MSSubClass', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond']:
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)
    
    # Ordinal features (these will be handled separately)
    ordinal_features = [
        'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
        'PoolQC', 'Fence'
    ]
    
    # Remove ordinal features from categorical list
    categorical_features = [f for f in categorical_features if f not in ordinal_features]
    
    # Log the categorization
    logger.info(f"\nNumeric features ({len(numeric_features)}): {', '.join(numeric_features[:5])}...")
    logger.info(f"Categorical features ({len(categorical_features)}): {', '.join(categorical_features[:5])}...")
    logger.info(f"Ordinal features ({len(ordinal_features)}): {', '.join(ordinal_features[:5])}...")
    
    return numeric_features, categorical_features, ordinal_features

def create_ordinal_mappings():
    """Create mappings for ordinal features."""
    quality_map = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    finish_map = {'NA': 0, 'No': 1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5, 'ALQ': 6, 'GLQ': 7}
    exposure_map = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    fence_map = {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    shape_map = {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
    
    ordinal_mappings = {
        'ExterQual': quality_map, 'ExterCond': quality_map,
        'BsmtQual': quality_map, 'BsmtCond': quality_map,
        'HeatingQC': quality_map, 'KitchenQual': quality_map,
        'FireplaceQu': quality_map, 'GarageQual': quality_map,
        'GarageCond': quality_map, 'PoolQC': quality_map,
        'BsmtFinType1': finish_map, 'BsmtFinType2': finish_map,
        'BsmtExposure': exposure_map,
        'Fence': fence_map,
        'LotShape': shape_map,
    }
    
    return ordinal_mappings

def preprocess_house_prices(train_df, test_df):
    """
    Preprocess Kaggle House Prices dataset with attention to potential biases.
    """
    # Initialize preprocessing summary
    preprocessing_summary = {
        'missing_values_handling': {
            'numeric': 'Median imputation (could underrepresent extreme values)',
            'categorical': 'Constant value imputation (adds "missing" category)',
            'ordinal': 'Mapped to lowest category ("NA" or equivalent)'
        },
        'scaling': 'StandardScaler (zero mean, unit variance)',
        'encoding': {
            'categorical': 'One-hot encoding with dropped first category',
            'ordinal': 'Mapped to numeric values preserving order'
        },
        'potential_biases': [
            'Median imputation might underrepresent minority neighborhoods',
            'Some quality metrics might reflect historical biases',
            'Neighborhood encoding might perpetuate existing market biases'
        ]
    }

    # Remove Id column and separate target
    train_y = train_df['SalePrice']
    train_df = train_df.drop(['Id', 'SalePrice'], axis=1)
    test_df = test_df.drop('Id', axis=1)
    
    def log_feature_info(df, feature_list, prefix=""):
        """Helper function to log feature information"""
        logger.info(f"\n{prefix} Features Summary:")
        for feature in feature_list:
            if feature in df.columns:
                unique_vals = df[feature].nunique()
                missing = df[feature].isnull().sum()
                logger.info(f"{feature}: {unique_vals} unique values, {missing} missing values")
    
    # Identify feature types
    numeric_features, categorical_features, ordinal_features = identify_features(train_df)
    
    # Remove ordinal features from categorical list
    categorical_features = [f for f in categorical_features if f not in ordinal_features]
    
    # Check for bias in missing values across neighborhoods
    # Calculate missing values by neighborhood
    missing_by_neighborhood = (train_df
        .drop('Neighborhood', axis=1)  # Exclude the grouping column
        .groupby(train_df['Neighborhood'])
        .apply(lambda x: x.isnull().sum())
    )
    logger.info(f"\nMissing values by neighborhood (top 5 features):\n{missing_by_neighborhood.mean().sort_values(ascending=False)[:5]}")
    
    # Log initial category distributions for potential mismatches
    for feature in categorical_features:
        if feature in train_df.columns:
            train_categories = set(train_df[feature].dropna().unique())
            test_categories = set(test_df[feature].dropna().unique())
            new_categories = test_categories - train_categories
            if new_categories:
                logger.info(f"\nFeature '{feature}' has {len(new_categories)} categories in test set "
                          f"not present in training: {new_categories}")

    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # Explicitly handle unknown categories with a warning
        ('onehot', OneHotEncoder(
            drop='first', 
            sparse_output=False, 
            handle_unknown='ignore',  # Will encode unknown categories as zeros
            min_frequency=0.01  # Only keep categories that appear in at least 1% of data
        ))
    ])
    
    # Handle ordinal features first
    ordinal_mappings = create_ordinal_mappings()
    
    # Apply ordinal transformations
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for feature, mapping in ordinal_mappings.items():
        if feature in train_df.columns:
            train_df[feature] = train_df[feature].fillna('NA').map(mapping)
            test_df[feature] = test_df[feature].fillna('NA').map(mapping)
    
    # Convert specified categorical features to string type
    for feature in categorical_features:
        if feature in train_df.columns:
            train_df[feature] = train_df[feature].astype(str)
            test_df[feature] = test_df[feature].astype(str)
    
    # Log detailed information about features before transformation
    log_feature_info(train_df, numeric_features, "Numeric")
    log_feature_info(train_df, categorical_features, "Categorical")
    
    # Fit and transform the data
    logger.info("\nStarting data transformation...")
            
    # Create preprocessing pipeline with verbose output
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False  # This helps prevent feature name issues
    )
    
    # Fit and transform training data
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    # Fit the preprocessor first
    X_train = preprocessor.fit_transform(train_df)
    
    # Now get the actual feature names from the fitted preprocessor
    feature_names = []
    
    # Add numeric feature names
    if numeric_features:
        feature_names.extend(numeric_features)
    
    # Add categorical feature names (only for features that were actually transformed)
    if categorical_features:
        cat_transformer = preprocessor.named_transformers_['cat']
        encoder = cat_transformer.named_steps['onehot']
        
        # Get the actual feature names from the encoder
        cat_features = []
        for i, (feature, cats) in enumerate(zip(categorical_features, encoder.categories_)):
            # Add all categories except the first one (which is dropped)
            for cat in cats[1:]:
                cat_features.append(f"{feature}_{cat}")
        
        feature_names.extend(cat_features)
    
    # Verify the shapes match before creating DataFrames
    if X_train.shape[1] != len(feature_names):
        logger.warning(f"Shape mismatch: got {X_train.shape[1]} features but {len(feature_names)} names")
        # Use generic feature names instead
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # Create the transformed DataFrames
    X_train_processed = pd.DataFrame(X_train, columns=feature_names)
    X_test_processed = pd.DataFrame(preprocessor.transform(test_df), columns=feature_names)
    
    # Log transformation results
    logger.info(f"\nFinal feature count: {X_train.shape[1]}")
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Generated feature names: {len(feature_names)}")
    
    # Document potential biases and preprocessing decisions
    # Document preprocessing decisions and category handling
    preprocessing_summary['category_handling'] = {
        'strategy': 'Rare categories (< 1% frequency) are grouped',
        'unknown_handling': 'New categories in test set encoded as zeros',
        'potential_impacts': [
            'Rare categories in neighborhoods might be overlooked',
            'New property types in test set will have zero encoding',
            'Consider if rare categories are disproportionately associated with any demographic group'
        ]
    }
    
    logger.info("\nPreprocessing Summary:")
    for key, value in preprocessing_summary.items():
        logger.info(f"\n{key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            for item in value:
                logger.info(f"  - {item}")
        else:
            logger.info(f"  {value}")
    
    return X_train_processed, X_test_processed, train_y, preprocessing_summary, preprocessor

# Example usage
if __name__ == "__main__":
    # Load data
    train_df, test_df = load_and_examine_data('train.csv', 'test.csv')
    
    # Preprocess data
    X_train, X_test, y_train, summary, preprocessor = preprocess_house_prices(train_df, test_df)
    
    print("\nShape of processed training data:", X_train.shape)
    print("Shape of processed test data:", X_test.shape)
    print("\nSample of processed features:", X_train.columns[:5].tolist())