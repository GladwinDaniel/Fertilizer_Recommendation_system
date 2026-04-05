import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def augment_data(df, num_samples=1000):
    """
    Augment the dataset by generating synthetic samples based on existing data distributions.
    """
    np.random.seed(42)
    
    # Encode categorical columns
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()
    
    df_copy = df.copy()
    df_copy['Soil Type'] = le_soil.fit_transform(df_copy['Soil Type'])
    df_copy['Crop Type'] = le_crop.fit_transform(df_copy['Crop Type'])
    df_copy['Fertilizer Name'] = le_fert.fit_transform(df_copy['Fertilizer Name'])
    
    # Generate synthetic data
    augmented_data = []
    for _ in range(num_samples):
        # Sample from existing distributions
        sample = {}
        for col in df_copy.columns:
            if col in ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']:
                # Add noise to numerical columns
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                sample[col] = np.random.normal(mean, std)
            else:
                # Sample from categorical
                sample[col] = np.random.choice(df_copy[col].unique())
        
        augmented_data.append(sample)
    
    augmented_df = pd.DataFrame(augmented_data)
    
    # Decode back
    augmented_df['Soil Type'] = le_soil.inverse_transform(augmented_df['Soil Type'].astype(int))
    augmented_df['Crop Type'] = le_crop.inverse_transform(augmented_df['Crop Type'].astype(int))
    augmented_df['Fertilizer Name'] = le_fert.inverse_transform(augmented_df['Fertilizer Name'].astype(int))
    
    return pd.concat([df, augmented_df], ignore_index=True)

if __name__ == "__main__":
    df = pd.read_csv('Fertilizer Prediction.csv')
    augmented_df = augment_data(df, 500)
    augmented_df.to_csv('augmented_dataset.csv', index=False)
    print(f"Original dataset: {len(df)} samples")
    print(f"Augmented dataset: {len(augmented_df)} samples")