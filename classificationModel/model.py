import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional
import cv2

class SandClassifier:
    """Classification model for beach sand type (Dune/Intertidal/Berm)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.classes = ['Dune', 'Intertidal', 'Berm']
        self.model = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: Input image array (preprocessed)
        Returns:
            Feature vector for classification
        """
        # Color features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_features = []
        
        # Mean and std of each channel
        for channel in cv2.split(hsv):
            color_features.extend([np.mean(channel), np.std(channel)])
        
        # Grain size distribution features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate grain properties
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
        
        if len(areas) > 0:
            size_features = [
                np.mean(areas),
                np.std(areas),
                np.percentile(areas, 25),
                np.percentile(areas, 75)
            ]
        else:
            size_features = [0, 0, 0, 0]
        
        # Combine all features
        features = np.array(color_features + size_features).reshape(1, -1)
        return features
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: Training features
            y: Training labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, image: np.ndarray) -> str:
        """
        Args:
            image: Input image array (preprocessed)
        Returns:
            Predicted sand class
        """
        # Extract features
        features = self.extract_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        return self.classes[prediction]
    
    def save_model(self, model_path: str) -> None:
        """
        Args:
            model_path: Path to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes
        }
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Args:
            model_path: Path to load the model from
        """
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.classes = model_data['classes']
