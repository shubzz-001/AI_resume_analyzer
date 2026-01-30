import pickle
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np

from ml.model_config import (
    JOB_ROLE_MODEL,
    MIN_CONFIDENCE_THRESHOLD,
    TOP_N_PREDICTIONS,
    DEFAULT_ROLE,
    get_recommended_roles
)

logger = logging.getLogger(__name__)

# Global model cache (singleton pattern)
_model_cache = None
_metadata_cache = None


def load_model():
    """
    Load or get cached model.
    """
    global _model_cache, _metadata_cache

    # Return cached model if available
    if _model_cache is not None:
        return _model_cache, _metadata_cache

    model_path = JOB_ROLE_MODEL

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please train the model first by running: python ml/train_model.py"
        )

    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Handle both old and new format
        if isinstance(model_data, dict):
            model = model_data['pipeline']
            metadata = model_data.get('metadata', {})
        else:
            # Old format (just the pipeline)
            model = model_data
            metadata = {}

        # Cache the model
        _model_cache = model
        _metadata_cache = metadata

        logger.info(f"Model loaded successfully from {model_path}")
        if metadata:
            logger.info(f"Model version: {metadata.get('version', 'unknown')}")
            logger.info(f"Model accuracy: {metadata.get('accuracy', 'unknown')}")

        return model, metadata

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


def predict_job_role(
        cleaned_resume_text: str,
        return_top_n: int = 1,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD
) -> Tuple[str, float]:
    """
    Predict job role from resume text.
    """
    # Validate input
    if not cleaned_resume_text or not isinstance(cleaned_resume_text, str):
        raise ValueError("Resume text must be a non-empty string")

    if len(cleaned_resume_text.strip()) < 10:
        raise ValueError("Resume text is too short for prediction")

    try:
        # Load model
        model, metadata = load_model()

        # Make prediction
        prediction = model.predict([cleaned_resume_text])[0]
        probabilities = model.predict_proba([cleaned_resume_text])[0]

        # Get confidence
        max_prob_idx = np.argmax(probabilities)
        confidence = probabilities[max_prob_idx]

        # Check confidence threshold
        if confidence < min_confidence:
            logger.warning(
                f"Low confidence prediction: {prediction} ({confidence:.2%}). "
                f"Returning default role."
            )
            return DEFAULT_ROLE, round(confidence * 100, 2)

        logger.info(f"Predicted role: {prediction} with {confidence:.2%} confidence")

        return prediction, round(confidence * 100, 2)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_top_n_roles(
        cleaned_resume_text: str,
        top_n: int = TOP_N_PREDICTIONS
) -> List[Dict[str, any]]:
    """
    Get top N job role predictions with probabilities.
    """
    # Validate input
    if not cleaned_resume_text or not isinstance(cleaned_resume_text, str):
        return []

    try:
        # Load model
        model, metadata = load_model()

        # Get all probabilities
        probabilities = model.predict_proba([cleaned_resume_text])[0]
        classes = model.classes_

        # Get top N
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        results = []
        for idx in top_indices:
            results.append({
                'role': classes[idx],
                'confidence': round(probabilities[idx] * 100, 2),
                'confidence_score': probabilities[idx]
            })

        logger.info(f"Top {top_n} predictions: {[r['role'] for r in results]}")

        return results

    except Exception as e:
        logger.error(f"Error getting top N predictions: {str(e)}")
        return []


def predict_with_explanation(
        cleaned_resume_text: str
) -> Dict[str, any]:
    """
    Predict job role with explanation and alternative suggestions.
    """
    try:
        # Get top predictions
        top_predictions = predict_top_n_roles(cleaned_resume_text, top_n=3)

        if not top_predictions:
            return {
                'predicted_role': DEFAULT_ROLE,
                'confidence': 0,
                'alternatives': [],
                'explanation': 'Could not determine role with confidence'
            }

        primary = top_predictions[0]

        # Create explanation
        if primary['confidence'] >= 70:
            explanation = f"Strong match for {primary['role']} based on resume content"
        elif primary['confidence'] >= 50:
            explanation = f"Moderate match for {primary['role']}"
        else:
            explanation = f"Weak match - consider alternative roles"

        return {
            'predicted_role': primary['role'],
            'confidence': primary['confidence'],
            'alternatives': top_predictions[1:],
            'explanation': explanation,
            'all_predictions': top_predictions
        }

    except Exception as e:
        logger.error(f"Error in predict_with_explanation: {str(e)}")
        return {
            'predicted_role': DEFAULT_ROLE,
            'confidence': 0,
            'alternatives': [],
            'explanation': f'Error: {str(e)}'
        }


def predict_from_skills(skills: List[str]) -> Dict[str, any]:
    """
    Predict job role based on skills alone (fallback method).
    """
    recommended_roles = get_recommended_roles(skills)

    if not recommended_roles:
        return {
            'predicted_role': DEFAULT_ROLE,
            'confidence': 0,
            'method': 'default'
        }

    return {
        'predicted_role': recommended_roles[0],
        'confidence': 60.0,  # Moderate confidence for skill-based
        'alternatives': recommended_roles[1:],
        'method': 'skill_based'
    }


def get_model_info() -> Dict[str, any]:
    """
    Get information about the loaded model.
    """
    try:
        model, metadata = load_model()

        return {
            'model_loaded': True,
            'model_path': str(JOB_ROLE_MODEL),
            'metadata': metadata,
            'classes': list(model.classes_) if hasattr(model, 'classes_') else []
        }
    except Exception as e:
        return {
            'model_loaded': False,
            'error': str(e)
        }


def validate_model() -> Tuple[bool, str]:
    """
    Validate that model is working correctly.
    """
    try:
        # Try to load model
        model, metadata = load_model()

        # Try a test prediction
        test_text = "experienced software engineer with python java and data structures"
        prediction, confidence = predict_job_role(test_text)

        if prediction and confidence > 0:
            return True, "Model is working correctly"
        else:
            return False, "Model returned invalid prediction"

    except Exception as e:
        return False, f"Model validation failed: {str(e)}"


class JobRolePredictor:
    """
    Class-based predictor for advanced usage.
    """

    def __init__(self):
        """Initialize predictor and load model."""
        self.model, self.metadata = load_model()
        self.classes = list(self.model.classes_)

    def predict(
            self,
            text: str,
            return_confidence: bool = True
    ) -> Tuple[str, float]:
        """
        Predict job role.
        """
        return predict_job_role(text)

    def predict_top_n(self, text: str, n: int = 3) -> List[Dict]:
        """
        Get top N predictions.
        """
        return predict_top_n_roles(text, n)

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predict multiple resumes at once.
        """
        results = []
        for text in texts:
            try:
                role, confidence = predict_job_role(text)
                results.append((role, confidence))
            except Exception as e:
                logger.error(f"Error predicting batch item: {str(e)}")
                results.append((DEFAULT_ROLE, 0.0))

        return results

    def get_feature_importance(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most important features for prediction.
        """
        try:
            # Get TF-IDF features
            tfidf = self.model.named_steps['tfidf']
            features = tfidf.transform([text])

            # Get feature names
            feature_names = tfidf.get_feature_names_out()

            # Get top features
            top_indices = features.toarray()[0].argsort()[-top_n:][::-1]

            return [(feature_names[i], features[0, i]) for i in top_indices]

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return []


# Backward compatibility
def get_prediction(cleaned_text: str) -> Tuple[str, float]:
    """
    Legacy function for backward compatibility.
    """
    return predict_job_role(cleaned_text)