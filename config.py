"""
Configuration settings for Resume Analyzer application.
"""
import os
from pathlib import Path

# PROJECT PATHS
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
ML_MODELS_DIR = ROOT_DIR / "ml" / "models"

# Data subdirectories
SKILLS_DIR = DATA_DIR / "skills"
JOBS_DIR = DATA_DIR / "jobs"
ATS_DIR = DATA_DIR / "ats"
TEMPLATES_DIR = DATA_DIR / "templates"

# ============================================
# MODEL CONFIGURATIONS
# ============================================
JOB_ROLE_MODEL_PATH = ML_MODELS_DIR / "job_role_model.pkl"

# Sentence Transformer Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CACHE_SIZE = 1000

# ============================================
# DATA FILES
# ============================================
# Skills databases
TECHNICAL_SKILLS_FILE = SKILLS_DIR / "technical_skills.json"
SOFT_SKILLS_FILE = SKILLS_DIR / "soft_skills.json"
DOMAIN_SKILLS_FILE = SKILLS_DIR / "domain_skills.json"
SKILLS_LIST_FILE = SKILLS_DIR / "skills_list.txt"

# Job data
JOB_DESCRIPTIONS_FILE = JOBS_DIR / "job_descriptions.csv"
JOB_TITLES_FILE = JOBS_DIR / "job_titles.json"

# ATS data
ATS_JOB_SKILLS_FILE = ATS_DIR / "ats_job_skills.json"
JOB_REQUIRED_SKILLS_FILE = ATS_DIR / "job_required_skills.json"
ACTION_VERBS_FILE = ATS_DIR / "action_verbs.json"

# ============================================
# SCORING THRESHOLDS
# ============================================
# Semantic similarity thresholds
SEMANTIC_SIMILARITY_THRESHOLD = 0.55
HIGH_SIMILARITY_THRESHOLD = 0.70
LOW_SIMILARITY_THRESHOLD = 0.40

# ATS scoring weights
KEYWORD_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# Resume score thresholds
EXCELLENT_SCORE = 80
GOOD_SCORE = 60
FAIR_SCORE = 40

# ============================================
# PROCESSING LIMITS
# ============================================
MIN_RESUME_LENGTH = 50  # characters
MAX_RESUME_LENGTH = 50000  # characters
MAX_FILE_SIZE_MB = 10

# ============================================
# UI SETTINGS
# ============================================
APP_TITLE = "AI Resume Analyzer Pro"
APP_ICON = "📄"
PAGE_LAYOUT = "wide"

# Supported file types
SUPPORTED_RESUME_FORMATS = ["pdf", "docx", "txt"]

# Display settings
MAX_JOBS_DISPLAY = 5
MAX_SKILLS_DISPLAY = 10
MAX_TIPS_DISPLAY = 5

# ============================================
# JOB ROLES
# ============================================
SUPPORTED_JOB_ROLES = [
    "Data Analyst",
    "ML Engineer",
    "Software Engineer",
    "Backend Developer",
    "Frontend Developer",
    "DevOps Engineer",
    "Full Stack Developer",
    "Data Scientist",
    "Product Manager"
]

DEFAULT_JOB_ROLE = "Software Engineer"

# ============================================
# FEATURE FLAGS
# ============================================
ENABLE_SEMANTIC_MATCHING = True
ENABLE_AI_SUGGESTIONS = True
ENABLE_EXPORT = True
ENABLE_TEMPLATE_BUILDER = False  # Phase 2
ENABLE_COVER_LETTER = False  # Phase 2

# ============================================
# LOGGING
# ============================================
LOG_LEVEL = "INFO"
LOG_FILE = ROOT_DIR / "app.log"

# ============================================
# CACHE SETTINGS
# ============================================
ENABLE_CACHE = True
CACHE_TTL = 3600  # seconds (1 hour)

# ============================================
# VALIDATION RULES
# ============================================
# Profile extraction patterns
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_PATTERN = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
LINKEDIN_PATTERN = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
GITHUB_PATTERN = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'

# Required resume sections
REQUIRED_SECTIONS = ["experience", "education", "skills"]
OPTIONAL_SECTIONS = ["projects", "certifications", "awards"]

# ============================================
# EXPORT SETTINGS
# ============================================
EXPORT_FORMATS = ["pdf", "docx", "txt", "json"]
DEFAULT_EXPORT_FORMAT = "pdf"

# ============================================
# ERROR MESSAGES
# ============================================
ERROR_FILE_TOO_LARGE = f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
ERROR_INVALID_FORMAT = f"Unsupported format. Please upload: {', '.join(SUPPORTED_RESUME_FORMATS)}"
ERROR_EMPTY_FILE = "The uploaded file is empty or corrupted"
ERROR_NO_TEXT = "Could not extract text from the file"
ERROR_MODEL_NOT_FOUND = "ML model not found. Please train the model first."

# ============================================
# SUCCESS MESSAGES
# ============================================
SUCCESS_UPLOAD = "✅ Resume uploaded successfully!"
SUCCESS_ANALYSIS = "✅ Analysis completed!"
SUCCESS_EXPORT = "✅ Resume exported successfully!"

# ============================================
# HELPER FUNCTIONS
# ============================================
def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, SKILLS_DIR, JOBS_DIR, ATS_DIR, TEMPLATES_DIR, ML_MODELS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_file_path(filename: str, subdirectory: str = "") -> Path:
    """Get absolute file path."""
    if subdirectory:
        return DATA_DIR / subdirectory / filename
    return DATA_DIR / filename

def validate_file_exists(filepath: Path) -> bool:
    """Check if file exists."""
    return filepath.exists() and filepath.is_file()

# Initialize directories on import
ensure_directories()