import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import cv2
from typing import Tuple, Optional

from model import SandClassifier

class ModelTrainer:
    """Trainer class for sand classification model with grid search and periodic retraining"""
    
    def __init__(self, model_save_dir: str):
        """
        Args:
            model_save_dir: Directory to save trained models
        """
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Define parameter grid for grid search
        self.param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    
    def prepare_data(self, image_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image_dir: Directory containing training images organized in class folders
        Returns:
            Tuple of (features, labels)
        """
        features = []
        labels = []
        classifier = SandClassifier()
        
        # Iterate through class directories
        for class_idx, class_name in enumerate(os.listdir(image_dir)):
            class_dir = os.path.join(image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Process each image in class directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                # Read and extract features
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                features.append(classifier.extract_features(image).flatten())
                labels.append(class_idx)
        
        return np.array(features), np.array(labels)
    
    def train_with_grid_search(self, X: np.ndarray, y: np.ndarray) -> SandClassifier:
        """
        Args:
            X: Training features
            y: Training labels
        Returns:
            Trained classifier with best parameters
        """
        # Create base classifier
        classifier = SandClassifier()
        
        # Set up grid search
        grid_search = GridSearchCV(
            classifier.model,
            self.param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit grid search
        X_scaled = classifier.scaler.fit_transform(X)
        grid_search.fit(X_scaled, y)
        
        # Update classifier with best model
        classifier.model = grid_search.best_estimator_
        
        return classifier
    
    def train_and_evaluate(self, image_dir: str, test_size: float = 0.2) -> Tuple[float, SandClassifier]:
        """
        Args:
            image_dir: Directory containing training images
            test_size: Fraction of data to use for testing
        Returns:
            Tuple of (accuracy, trained_classifier)
        """
        # Prepare data
        X, y = self.prepare_data(image_dir)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model with grid search
        classifier = self.train_with_grid_search(X_train, y_train)
        
        # Evaluate
        X_test_scaled = classifier.scaler.transform(X_test)
        accuracy = classifier.model.score(X_test_scaled, y_test)
        
        return accuracy, classifier
    
    def save_model(self, classifier: SandClassifier, accuracy: float) -> str:
        """
        Args:
            classifier: Trained classifier to save
            accuracy: Model accuracy
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"sand_classifier_{timestamp}_acc{accuracy:.3f}.joblib"
        model_path = os.path.join(self.model_save_dir, model_name)
        
        classifier.save_model(model_path)
        return model_path
    
    def retrain_if_needed(self, image_dir: str, accuracy_threshold: float = 0.85) -> Optional[str]:
        """
        Args:
            image_dir: Directory containing training images
            accuracy_threshold: Minimum accuracy threshold for retraining
        Returns:
            Path to new model if retrained, None otherwise
        """
        # Train and evaluate
        accuracy, classifier = self.train_and_evaluate(image_dir)
        
        # Save if accuracy improved
        if accuracy >= accuracy_threshold:
            return self.save_model(classifier, accuracy)
        
        return None
