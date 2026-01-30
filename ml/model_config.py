from pathlib import Path
from config import ML_MODELS_DIR, JOB_ROLE_MODEL_PATH, JOBS_DIR

# MODEL PATHS
JOB_ROLE_MODEL = JOB_ROLE_MODEL_PATH
SKILL_CLASSIFIER_MODEL = ML_MODELS_DIR / "skill_classifier.pkl"

# TRAINING DATA
TRAINING_DATA_FILE = JOBS_DIR / "job_descriptions.csv"

# ============================================
# MODEL PARAMETERS
# ============================================

# TF-IDF Vectorizer settings
TFIDF_CONFIG = {
    'max_features': 3000,
    'ngram_range': (1, 2),  # Unigrams and bigrams
    'min_df': 2,  # Minimum document frequency
    'max_df': 0.8,  # Maximum document frequency
    'sublinear_tf': True  # Use sublinear term frequency scaling
}

# Logistic Regression settings
LOGISTIC_REGRESSION_CONFIG = {
    'max_iter': 1000,
    'multi_class': 'multinomial',
    'solver': 'lbfgs',
    'random_state': 42,
    'class_weight': 'balanced'  # Handle imbalanced classes
}

# Random Forest settings (alternative classifier)
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================
# PREDICTION SETTINGS
# ============================================
MIN_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to accept prediction
TOP_N_PREDICTIONS = 3  # Number of top predictions to return

# ============================================
# JOB ROLES
# ============================================
JOB_ROLES = [
    "Data Analyst",
    "ML Engineer",
    "Software Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "DevOps Engineer",
    "Data Scientist",
    "Product Manager",
    "Mobile Developer"
]

DEFAULT_ROLE = "Software Engineer"

# ============================================
# TRAINING SETTINGS
# ============================================
TEST_SIZE = 0.2  # Train/test split ratio
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# ============================================
# MODEL EVALUATION METRICS
# ============================================
METRICS_TO_TRACK = [
    'accuracy',
    'precision',
    'recall',
    'f1_score'
]

# ============================================
# MODEL VERSIONING
# ============================================
MODEL_VERSION = "1.0.0"
MODEL_METADATA = {
    'version': MODEL_VERSION,
    'algorithm': 'LogisticRegression',
    'feature_extraction': 'TF-IDF',
    'date_created': None,  # Will be set during training
    'training_samples': None,  # Will be set during training
    'accuracy': None  # Will be set during training
}

# ============================================
# FEATURE ENGINEERING
# ============================================
# Important keywords for each job role
ROLE_KEYWORDS = {
    "Data Analyst": [
        'sql', 'excel', 'tableau', 'power bi', 'data visualization',
        'statistics', 'python', 'r', 'data analysis', 'reporting'
    ],
    "ML Engineer": [
        'machine learning', 'deep learning', 'tensorflow', 'pytorch',
        'neural networks', 'nlp', 'computer vision', 'feature engineering',
        'model deployment', 'mlops'
    ],
    "Software Engineer": [
        'java', 'python', 'c++', 'data structures', 'algorithms',
        'object oriented', 'design patterns', 'git', 'testing', 'debugging'
    ],
    "Backend Developer": [
        'api', 'rest', 'database', 'sql', 'nosql', 'microservices',
        'spring boot', 'django', 'flask', 'node.js', 'authentication'
    ],
    "Frontend Developer": [
        'html', 'css', 'javascript', 'react', 'vue', 'angular',
        'responsive design', 'ui/ux', 'typescript', 'webpack'
    ],
    "Full Stack Developer": [
        'frontend', 'backend', 'full stack', 'react', 'node.js',
        'database', 'api', 'deployment', 'git', 'agile'
    ],
    "DevOps Engineer": [
        'docker', 'kubernetes', 'ci/cd', 'jenkins', 'aws', 'azure',
        'terraform', 'ansible', 'linux', 'monitoring', 'automation'
    ],
    "Data Scientist": [
        'machine learning', 'statistics', 'python', 'r', 'pandas',
        'numpy', 'scikit-learn', 'data analysis', 'visualization',
        'hypothesis testing', 'a/b testing'
    ],
    "Product Manager": [
        'product management', 'roadmap', 'agile', 'scrum', 'stakeholder',
        'user research', 'analytics', 'prioritization', 'requirements',
        'product strategy'
    ],
    "Mobile Developer": [
        'android', 'ios', 'swift', 'kotlin', 'react native', 'flutter',
        'mobile ui', 'app store', 'google play', 'mobile development'
    ]
}

# ============================================
# VALIDATION SETTINGS
# ============================================
# Minimum samples per class for training
MIN_SAMPLES_PER_CLASS = 5

# Minimum text length for valid job description
MIN_JOB_DESC_LENGTH = 50

# ============================================
# LOGGING
# ============================================
LOG_PREDICTIONS = True
LOG_TRAINING_METRICS = True


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_model_info() -> dict:

    return {
        'model_path': str(JOB_ROLE_MODEL),
        'tfidf_config': TFIDF_CONFIG,
        'classifier_config': LOGISTIC_REGRESSION_CONFIG,
        'job_roles': JOB_ROLES,
        'metadata': MODEL_METADATA
    }


def validate_training_data(df) -> tuple:

    required_columns = ['job_description', 'job_title']

    # Check columns
    if not all(col in df.columns for col in required_columns):
        return False, f"Missing required columns: {required_columns}"

    # Check for empty data
    if len(df) == 0:
        return False, "No training data provided"

    # Check minimum samples per class
    class_counts = df['job_title'].value_counts()
    insufficient = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]

    if len(insufficient) > 0:
        return False, f"Insufficient samples for classes: {insufficient.index.tolist()}"

    # Check text length
    short_texts = df[df['job_description'].str.len() < MIN_JOB_DESC_LENGTH]
    if len(short_texts) > 0:
        return False, f"Found {len(short_texts)} job descriptions that are too short"

    return True, "Valid"


def get_recommended_roles(skills: list) -> list:

    skill_set = set(s.lower() for s in skills)
    role_scores = {}

    for role, keywords in ROLE_KEYWORDS.items():
        # Count matching keywords
        matches = sum(1 for kw in keywords if kw in skill_set)
        if matches > 0:
            role_scores[role] = matches

    # Sort by score
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)

    return [role for role, score in sorted_roles[:TOP_N_PREDICTIONS]]