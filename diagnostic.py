"""
Comprehensive Project Diagnostic and Testing Script
Tests all components of the Personalized Learning RL project
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import traceback
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'
BOLD = '\033[1m'

class Diagnostic:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        # Open log file with UTF-8 encoding to handle emojis
        self.log_file = open('diagnostic_log.txt', 'w', encoding='utf-8')
        
    def log(self, message, level='INFO'):
        """Log message to both console and file"""
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] [{level}] {message}"
        print(log_msg)
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()
    
    def test_passed(self, test_name, message):
        """Record passed test"""
        self.results['passed'].append(test_name)
        self.log(f"✅ PASS: {test_name} - {message}", 'INFO')
        print(f"{GREEN}✅ PASS{END}: {test_name} - {message}\n")
    
    def test_failed(self, test_name, error, exception=None):
        """Record failed test"""
        self.results['failed'].append(test_name)
        self.log(f"❌ FAIL: {test_name} - {error}", 'ERROR')
        print(f"{RED}❌ FAIL{END}: {test_name} - {error}")
        if exception:
            self.log(f"Exception: {str(exception)}", 'ERROR')
            print(f"   Exception: {str(exception)}")
        print()
    
    def test_warning(self, test_name, message):
        """Record warning"""
        self.results['warnings'].append(test_name)
        self.log(f"⚠️  WARNING: {test_name} - {message}", 'WARNING')
        print(f"{YELLOW}⚠️  WARNING{END}: {test_name} - {message}\n")

def test_directory_structure(diag):
    """Test if all required directories and files exist"""
    print(f"\n{BOLD}{'='*70}")
    print(f"1. TESTING DIRECTORY STRUCTURE")
    print(f"{'='*70}{END}\n")
    
    required_dirs = [
        'data',
        'models',
        'environment',
        'agents',
        'trainings',
        'plots',
        'webapp'
    ]
    
    required_files = [
        'data/questions.csv',
        'data/cleaned_df.csv',
        'curriculum_skills.json',
        'requirements.txt',
        'demo_session.py',
        'simple_quiz.py'
    ]
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            diag.test_passed(f"Directory: {dir_name}", f"Found at {os.path.abspath(dir_name)}")
        else:
            diag.test_failed(f"Directory: {dir_name}", f"Not found at {os.path.abspath(dir_name)}")
    
    # Check files
    for file_name in required_files:
        if os.path.isfile(file_name):
            file_size = os.path.getsize(file_name)
            diag.test_passed(f"File: {file_name}", f"Found ({file_size:,} bytes)")
        else:
            diag.test_failed(f"File: {file_name}", f"Not found")
    
    return len(diag.results['failed']) == 0

def test_data_files(diag):
    """Test data files integrity and content"""
    print(f"\n{BOLD}{'='*70}")
    print(f"2. TESTING DATA FILES")
    print(f"{'='*70}{END}\n")
    
    # Test questions.csv
    try:
        questions_df = pd.read_csv('data/questions.csv')
        num_rows = len(questions_df)
        num_cols = len(questions_df.columns)
        skills = questions_df['skill'].nunique() if 'skill' in questions_df.columns else 0
        
        diag.test_passed(
            "File: data/questions.csv",
            f"Loaded successfully - {num_rows} questions, {num_cols} columns, {skills} unique skills"
        )
        diag.log(f"Questions CSV columns: {list(questions_df.columns)}", 'DEBUG')
        
    except Exception as e:
        diag.test_failed("File: data/questions.csv", "Failed to load", e)
        return False
    
    # Test cleaned_df.csv
    try:
        cleaned_df = pd.read_csv('data/cleaned_df.csv')
        diag.test_passed(
            "File: data/cleaned_df.csv",
            f"Loaded successfully - {len(cleaned_df):,} rows"
        )
    except Exception as e:
        diag.test_failed("File: data/cleaned_df.csv", "Failed to load", e)
    
    # Test curriculum_skills.json
    try:
        with open('curriculum_skills.json', 'r') as f:
            skills_data = json.load(f)
        if isinstance(skills_data, list):
            diag.test_passed(
                "File: curriculum_skills.json",
                f"Loaded successfully - {len(skills_data)} skills defined"
            )
        else:
            diag.test_warning(
                "File: curriculum_skills.json",
                "Loaded but not a list - unexpected format"
            )
    except Exception as e:
        diag.test_failed("File: curriculum_skills.json", "Failed to load", e)
    
    return True

def test_model_files(diag):
    """Test if model files exist and can be inspected"""
    print(f"\n{BOLD}{'='*70}")
    print(f"3. TESTING MODEL FILES")
    print(f"{'='*70}{END}\n")
    
    model_dir = 'models'
    model_files = []
    
    if os.path.isdir(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(('.keras', '.h5')):
                file_path = os.path.join(model_dir, file)
                file_size = os.path.getsize(file_path)
                model_files.append((file, file_size))
                diag.log(f"Found model: {file} ({file_size:,} bytes)", 'DEBUG')
    
    if model_files:
        diag.test_passed(
            "Models directory contents",
            f"Found {len(model_files)} model files"
        )
        for model_name, size in model_files:
            print(f"   - {model_name}: {size:,} bytes ({size / (1024*1024):.2f} MB)")
    else:
        diag.test_failed(
            "Models directory contents",
            "No model files found (.keras or .h5)"
        )
        return False
    
    # Try to load models with TensorFlow
    try:
        import tensorflow as tf
        print(f"\n{BLUE}Attempting to load models with TensorFlow...{END}\n")
        
        models_loaded = 0
        models_failed = 0
        
        for model_name, size in model_files:
            model_path = os.path.join(model_dir, model_name)
            try:
                model = tf.keras.models.load_model(model_path)
                diag.test_passed(
                    f"Load model: {model_name}",
                    f"Successfully loaded - {model.count_params():,} parameters"
                )
                models_loaded += 1
                
                # Print model summary
                print(f"\n{BLUE}Model Architecture: {model_name}{END}")
                model.summary()
                print()
                
            except Exception as e:
                diag.test_warning(
                    f"Load model: {model_name}",
                    f"Failed to load - {str(e)[:100]}"
                )
                models_failed += 1
        
        diag.log(f"Models loaded: {models_loaded}/{len(model_files)}", 'SUMMARY')
        
        if models_loaded > 0:
            return True
        else:
            return False
            
    except ImportError:
        diag.test_failed(
            "TensorFlow import",
            "TensorFlow not available for model testing"
        )
        return False

def test_environment(diag):
    """Test if the RL environment can be imported and initialized"""
    print(f"\n{BOLD}{'='*70}")
    print(f"4. TESTING RL ENVIRONMENT")
    print(f"{'='*70}{END}\n")
    
    try:
        from environment.question_selection_env_main import QuestionSelectionEnv
        diag.test_passed(
            "Import QuestionSelectionEnv",
            "Successfully imported from environment.question_selection_env_main"
        )
    except ImportError as e:
        diag.test_failed(
            "Import QuestionSelectionEnv",
            "Failed to import environment",
            e
        )
        return False
    
    # Try to initialize environment
    try:
        questions_df = pd.read_csv('data/questions.csv')
        
        print(f"{BLUE}Initializing QuestionSelectionEnv...{END}\n")
        env = QuestionSelectionEnv(
            questions_df,
            max_steps=10,
            action_types=['skill', 'type']
        )
        
        diag.test_passed(
            "Initialize QuestionSelectionEnv",
            f"Environment initialized with {env.num_skills} skills, {env.num_question_types} question types"
        )
        diag.log(f"Environment action space: {env.action_space}", 'DEBUG')
        diag.log(f"Environment observation shape: {env.observation_space}", 'DEBUG')
        
        return True
        
    except Exception as e:
        diag.test_failed(
            "Initialize QuestionSelectionEnv",
            "Failed to initialize environment",
            e
        )
        diag.log(f"Traceback: {traceback.format_exc()}", 'ERROR')
        return False

def test_stable_baselines3(diag):
    """Test if Stable Baselines3 (PPO) is available"""
    print(f"\n{BOLD}{'='*70}")
    print(f"5. TESTING REINFORCEMENT LEARNING (PPO)")
    print(f"{'='*70}{END}\n")
    
    try:
        from stable_baselines3 import PPO
        diag.test_passed(
            "Import stable_baselines3.PPO",
            "PPO successfully imported"
        )
    except ImportError as e:
        diag.test_failed(
            "Import stable_baselines3.PPO",
            "Failed to import PPO",
            e
        )
        return False
    
    # Check for saved PPO models
    trainings_dir = 'trainings'
    ppo_dirs = [d for d in os.listdir(trainings_dir) if d.startswith('PPO_')]
    
    if ppo_dirs:
        diag.test_passed(
            "PPO training directories",
            f"Found {len(ppo_dirs)} PPO training runs"
        )
        print(f"   Training runs: {', '.join(sorted(ppo_dirs))}\n")
    else:
        diag.test_warning(
            "PPO training directories",
            "No PPO training directories found"
        )
    
    # Check for saved models in models/teacher
    teacher_dir = 'models/teacher'
    if os.path.isdir(teacher_dir):
        teacher_models = os.listdir(teacher_dir)
        if teacher_models:
            diag.test_passed(
                "Saved teacher models",
                f"Found {len(teacher_models)} teacher models"
            )
        else:
            diag.test_warning(
                "Saved teacher models",
                "Teacher directory exists but is empty"
            )
    else:
        diag.test_warning(
            "Saved teacher models",
            "models/teacher directory doesn't exist"
        )
    
    return True

def test_dependencies(diag):
    """Test if all required dependencies are installed"""
    print(f"\n{BOLD}{'='*70}")
    print(f"6. TESTING DEPENDENCIES")
    print(f"{'='*70}{END}\n")
    
    dependencies = {
        'tensorflow': 'TensorFlow (Deep Learning)',
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'stable_baselines3': 'Stable Baselines3 (RL)',
        'gymnasium': 'Gymnasium (RL environments)',
        'pandas': 'Pandas (Data)',
        'numpy': 'NumPy (Numerical)',
        'sklearn': 'Scikit-learn (ML)',
        'matplotlib': 'Matplotlib (Plotting)'
    }
    
    all_installed = True
    for module_name, display_name in dependencies.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            diag.test_passed(
                f"Package: {display_name}",
                f"Installed (version: {version})"
            )
        except ImportError:
            diag.test_failed(
                f"Package: {display_name}",
                "Not installed"
            )
            all_installed = False
    
    return all_installed

def generate_report(diag):
    """Generate final diagnostic report"""
    print(f"\n\n{BOLD}{'='*70}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*70}{END}\n")
    
    total_passed = len(diag.results['passed'])
    total_failed = len(diag.results['failed'])
    total_warnings = len(diag.results['warnings'])
    
    print(f"{GREEN}✅ Passed: {total_passed}{END}")
    print(f"{RED}❌ Failed: {total_failed}{END}")
    print(f"{YELLOW}⚠️  Warnings: {total_warnings}{END}\n")
    
    if total_failed == 0:
        print(f"{GREEN}{BOLD}🎉 ALL TESTS PASSED!{END}")
        print(f"Your project is ready to use.\n")
    else:
        print(f"{RED}{BOLD}⚠️  SOME TESTS FAILED{END}")
        print(f"Please review the errors above.\n")
    
    # Write report to file
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'passed': len(diag.results['passed']),
        'failed': len(diag.results['failed']),
        'warnings': len(diag.results['warnings']),
        'passed_tests': diag.results['passed'],
        'failed_tests': diag.results['failed'],
        'warning_tests': diag.results['warnings']
    }
    
    with open('diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    diag.log(f"Diagnostic report saved to diagnostic_report.json", 'INFO')
    diag.log_file.close()
    
    return total_failed == 0

def main():
    """Run all diagnostics"""
    print(f"\n{BOLD}{BLUE}{'='*70}")
    print(f"PERSONALIZED LEARNING RL - PROJECT DIAGNOSTIC")
    print(f"{'='*70}{END}\n")
    
    diag = Diagnostic()
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_directory_structure(diag)
    all_passed &= test_data_files(diag)
    all_passed &= test_model_files(diag)
    all_passed &= test_environment(diag)
    all_passed &= test_stable_baselines3(diag)
    all_passed &= test_dependencies(diag)
    
    # Generate report
    report_ok = generate_report(diag)
    
    print(f"\n{BLUE}Diagnostic files created:{END}")
    print(f"  - diagnostic_log.txt (detailed log)")
    print(f"  - diagnostic_report.json (summary report)\n")
    
    return 0 if report_ok else 1

if __name__ == '__main__':
    sys.exit(main())
