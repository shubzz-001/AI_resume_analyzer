import pandas as pd
import pickle
import logging
from datetime import datetime
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

from ml.model_config import (
    TRAINING_DATA_FILE,
    JOB_ROLE_MODEL,
    TFIDF_CONFIG,
    LOGISTIC_REGRESSION_CONFIG,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_METADATA,
    validate_training_data,
    CROSS_VALIDATION_FOLDS
)
from config import ensure_directories

logger = logging.getLogger(__name__)

# Ensure directories exist
ensure_directories()


def load_training_data(file_path: Path = TRAINING_DATA_FILE) -> pd.DataFrame:
    """
    Load training data from CSV file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Training data not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} training samples from {file_path}")

        # Validate data
        is_valid, message = validate_training_data(df)
        if not is_valid:
            raise ValueError(f"Invalid training data: {message}")

        return df

    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        raise


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess training data.
    """
    # Remove duplicates
    df = df.drop_duplicates(subset=['job_description'])

    # Remove NaN values
    df = df.dropna(subset=['job_description', 'job_title'])

    # Clean text (basic)
    df['job_description'] = df['job_description'].str.lower()
    df['job_description'] = df['job_description'].str.strip()

    X = df['job_description'].values
    y = df['job_title'].values

    logger.info(f"Preprocessed data: {len(X)} samples, {len(set(y))} classes")

    return X, y


def create_pipeline() -> Pipeline:
    """
    Create ML pipeline with TF-IDF and Logistic Regression.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_CONFIG)),
        ('clf', LogisticRegression(**LOGISTIC_REGRESSION_CONFIG))
    ])

    logger.info("Created ML pipeline: TF-IDF + Logistic Regression")
    return pipeline


def train_model(
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        save_path: Path = JOB_ROLE_MODEL
) -> Pipeline:
    """
    Train job role prediction model.
    """
    logger.info("Starting model training...")

    # Create pipeline
    pipeline = create_pipeline()

    # Train model
    pipeline.fit(X_train, y_train)
    logger.info("Model training completed")

    # Evaluate on training data
    train_accuracy = pipeline.score(X_train, y_train)
    logger.info(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test data if provided
    if X_test is not None and y_test is not None:
        test_accuracy = pipeline.score(X_test, y_test)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

        # Detailed evaluation
        y_pred = pipeline.predict(X_test)

        # Classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

    # Cross-validation
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=CROSS_VALIDATION_FOLDS
    )
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Save model
    save_model(pipeline, save_path, train_accuracy, len(X_train))

    return pipeline


def save_model(
        model: Pipeline,
        save_path: Path,
        accuracy: float,
        num_samples: int
) -> None:
    """
    Save trained model to disk.
    """
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Update metadata
    metadata = MODEL_METADATA.copy()
    metadata['date_created'] = datetime.now().isoformat()
    metadata['training_samples'] = num_samples
    metadata['accuracy'] = round(accuracy, 4)

    # Save model with metadata
    model_data = {
        'pipeline': model,
        'metadata': metadata
    }

    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Model metadata: {metadata}")


def load_model(model_path: Path = JOB_ROLE_MODEL) -> tuple:
    """
    Load trained model from disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Handle both old and new format
        if isinstance(model_data, dict):
            pipeline = model_data['pipeline']
            metadata = model_data.get('metadata', {})
        else:
            # Old format (just the pipeline)
            pipeline = model_data
            metadata = {}

        logger.info(f"Model loaded from: {model_path}")

        return pipeline, metadata

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def evaluate_model(model: Pipeline, X_test, y_test) -> dict:
    """
    Evaluate model performance.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return metrics


def train_from_scratch(data_file: Path = TRAINING_DATA_FILE) -> Pipeline:
    """
    Complete training workflow from scratch.
    """
    print("=" * 60)
    print("JOB ROLE PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n1. Loading training data...")
    df = load_training_data(data_file)
    print(f"   ✓ Loaded {len(df)} samples")
    print(f"   ✓ Classes: {df['job_title'].nunique()}")
    print(f"   ✓ Class distribution:\n{df['job_title'].value_counts()}")

    # Preprocess
    print("\n2. Preprocessing data...")
    X, y = preprocess_data(df)
    print(f"   ✓ Final dataset: {len(X)} samples")

    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")

    # Train
    print("\n4. Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    print("   ✓ Training completed")

    # Final evaluation
    print("\n5. Final evaluation...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"   ✓ Accuracy: {metrics['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return model


# Command-line interface
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Check if custom data file provided
        if len(sys.argv) > 1:
            data_file = Path(sys.argv[1])
        else:
            data_file = TRAINING_DATA_FILE

        # Train model
        model = train_from_scratch(data_file)

        print(f"\nModel saved to: {JOB_ROLE_MODEL}")
        print("You can now use the model for predictions!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)