# Adaptive Math Quiz Web App

This is a Streamlit-based web interface for the personalized learning RL project. It uses pre-trained models to deliver an adaptive quiz experience.

## Features
- Adaptive question selection based on student performance
- Multiple question types (multiple choice, fill-in)
- Real-time feedback
- Progress tracking
- Topic and difficulty visualization

## Setup

1. Create a virtual environment:
```cmd
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install requirements:
```cmd
pip install -r requirements.txt
```

3. Run the app:
```cmd
streamlit run adaptive_quiz_app.py
```

The app will open in your default web browser.

## How it Works

The app uses:
- Pre-trained DKT (Deep Knowledge Tracing) model to track student knowledge
- Reinforcement Learning environment for adaptive question selection
- Streamlit for the web interface

The system adapts to:
- Student performance
- Topic mastery
- Question difficulty

## Note
This is a simplified demo version. The full system can be extended with:
- More sophisticated question selection strategies
- Detailed analytics
- Student progress persistence
- Custom question sets