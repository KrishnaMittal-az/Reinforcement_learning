import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import from main project
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from environment.question_selection_env import QuestionSelectionEnv

class AdaptiveQuiz:
    def __init__(self):
        # Load questions
        self.questions_df = pd.read_csv(os.path.join(parent_dir, 'data', 'questions.csv'))
        
        # Initialize environment with pre-trained model
        self.env = QuestionSelectionEnv(
            questions_df=self.questions_df,
            lstm_model_path=os.path.join(parent_dir, 'models', 'dkt_model_pretrained_seq50.keras'),
            max_steps=10,  # Limit quiz length
            w_answerability=50,
            w_improvement=100,
            w_coverage=0.5
        )
        
        # Reset environment to start fresh
        self.observation = self.env.reset()
        
    def get_next_question(self, previous_correct=None):
        if previous_correct is not None:
            # Step the environment with the student's answer
            action = self.env.action_space.sample()  # For demo, we'll use random action
            obs, reward, done, info = self.env.step(action)
            
        # Get current question based on environment state
        question_info = self.env.current_question
        return question_info

def main():
    st.title("Adaptive Math Quiz")
    
    # Initialize session state
    if 'quiz' not in st.session_state:
        st.session_state.quiz = AdaptiveQuiz()
        st.session_state.current_question = None
        st.session_state.question_number = 0
        st.session_state.correct_answers = 0
        
    # Get next question if needed
    if st.session_state.current_question is None:
        st.session_state.current_question = st.session_state.quiz.get_next_question()
        st.session_state.question_number += 1
        
    # Display question
    st.subheader(f"Question {st.session_state.question_number}")
    question = st.session_state.current_question
    
    st.write(question['question_text'])
    
    # Handle different question types
    user_answer = None
    if question['question_type'] == 'multiple_choice' and question['choices']:
        choices = question['choices'].split('|')
        user_answer = st.radio("Select your answer:", choices)
    else:
        user_answer = st.text_input("Your answer:")
        
    # Submit button
    if st.button("Submit Answer"):
        # Check answer
        is_correct = str(user_answer).strip().lower() == str(question['answer']).strip().lower()
        
        if is_correct:
            st.success("Correct! 🎉")
            st.session_state.correct_answers += 1
        else:
            st.error(f"Not quite. The correct answer was: {question['answer']}")
            
        # Display progress
        st.write(f"Score: {st.session_state.correct_answers}/{st.session_state.question_number}")
        
        # Get next question
        st.session_state.current_question = st.session_state.quiz.get_next_question(is_correct)
        st.experimental_rerun()
        
    # Show skill being tested
    st.sidebar.subheader("Current Topic")
    st.sidebar.write(question['skill'])
    
    # Show difficulty
    st.sidebar.subheader("Difficulty")
    st.sidebar.write(question['difficulty'])

if __name__ == "__main__":
    main()