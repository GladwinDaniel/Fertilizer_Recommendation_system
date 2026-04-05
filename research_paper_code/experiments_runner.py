import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FertilizerRecommender:
    def __init__(self):
        self.models = {}
        self.best_model = None
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the dataset."""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Encode categorical variables
        self.le_soil = LabelEncoder()
        self.le_crop = LabelEncoder()
        self.le_fert = LabelEncoder()
        
        df['Soil Type'] = self.le_soil.fit_transform(df['Soil Type'])
        df['Crop Type'] = self.le_crop.fit_transform(df['Crop Type'])
        df['Fertilizer Name'] = self.le_fert.fit_transform(df['Fertilizer Name'])
        
        # Features and target
        X = df.drop('Fertilizer Name', axis=1)
        y = df['Fertilizer Name']
        
        return X, y
    
    def create_ensemble_model(self):
        """Create an ensemble model with multiple classifiers."""
        rf = RandomForestClassifier(random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        svc = SVC(probability=True, random_state=42)
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svc', svc)],
            voting='soft'
        )
        
        return ensemble
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Random Forest."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def train_models(self, X_train, y_train):
        """Train multiple models for comparison."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Individual models
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
        self.models['SVM'] = SVC(random_state=42)
        
        # Tuned model
        self.models['Tuned RF'] = self.hyperparameter_tuning(X_train, y_train)
        
        # Ensemble
        self.models['Ensemble'] = self.create_ensemble_model()
        
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            logger.info(f"Trained {name}")
        
        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and return results."""
        scaler = joblib.load('scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
        
        return results
    
    def plot_results(self, results):
        """Plot model comparison results."""
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=models, y=accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.show()

if __name__ == "__main__":
    recommender = FertilizerRecommender()
    
    # Load data
    X, y = recommender.load_and_preprocess_data('Fertilizer Prediction.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    recommender.train_models(X_train, y_train)
    
    # Evaluate
    results = recommender.evaluate_models(X_test, y_test)
    
    # Print results
    for model, metrics in results.items():
        print(f"{model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # Plot
    recommender.plot_results(results)
    
    # Save best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    joblib.dump(recommender.models[best_model_name], 'best_model.pkl')
    logger.info(f"Saved best model: {best_model_name}")