from ml.predictor import (
    predict_job_role,
    predict_top_n_roles,
    predict_with_explanation,
    predict_from_skills,
    get_model_info,
    validate_model,
    JobRolePredictor
)

from ml.train_model import (
    train_from_scratch,
    load_training_data,
    train_model,
    save_model,
    load_model,
    evaluate_model
)

from ml.model_config import (
    JOB_ROLES,
    DEFAULT_ROLE,
    ROLE_KEYWORDS,
    get_recommended_roles
)

__all__ = [
    # Predictor
    'predict_job_role',
    'predict_top_n_roles',
    'predict_with_explanation',
    'predict_from_skills',
    'get_model_info',
    'validate_model',
    'JobRolePredictor',

    # Training
    'train_from_scratch',
    'load_training_data',
    'train_model',
    'save_model',
    'load_model',
    'evaluate_model',

    # Config
    'JOB_ROLES',
    'DEFAULT_ROLE',
    'ROLE_KEYWORDS',
    'get_recommended_roles'
]


# Convenience functions

def quick_predict(resume_text: str) -> dict:

    return predict_with_explanation(resume_text)


def check_model_status() -> dict:

    is_valid, message = validate_model()
    info = get_model_info()

    return {
        'is_valid': is_valid,
        'message': message,
        'model_info': info
    }


def get_supported_roles() -> list:

    return JOB_ROLES.copy()