import os
import argparse
from preProcess.preprocess import PreProcessor
from classificationModel.model import SandClassifier
from classificationModel.train import ModelTrainer

def main():
    """Main execution function for sand grain analysis pipeline"""
    parser = argparse.ArgumentParser(description='Sand Grain Analysis Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                      help='Operation mode: train or predict')
    parser.add_argument('--input', required=True,
                      help='Input image path for predict mode or training data directory for train mode')
    parser.add_argument('--model-dir', default='classificationModel/weights',
                      help='Directory for model weights')
    parser.add_argument('--accuracy-threshold', type=float, default=0.85,
                      help='Accuracy threshold for model retraining')
    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = PreProcessor()

    if args.mode == 'train':
        # Training mode
        trainer = ModelTrainer(args.model_dir)
        new_model_path = trainer.retrain_if_needed(
            args.input,
            accuracy_threshold=args.accuracy_threshold
        )
        
        if new_model_path:
            print(f"New model trained and saved to: {new_model_path}")
        else:
            print("Model training did not meet accuracy threshold")

    else:
        # Prediction mode
        # Load the latest model
        model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.joblib')]
        if not model_files:
            raise ValueError("No trained model found")
            
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(args.model_dir, latest_model)
        classifier = SandClassifier(model_path)

        # Process image and predict
        try:
            # Preprocess image
            image, scale_factor = preprocessor.process(args.input)
            
            # Classify
            prediction = classifier.predict(image)
            
            print(f"Predicted sand type: {prediction}")
            print(f"Scale factor: {scale_factor:.4f} mm/pixel")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()
