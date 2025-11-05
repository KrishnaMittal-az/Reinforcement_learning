import gymnasium as gym
from stable_baselines3 import PPO
from environment.question_selection_env import QuestionSelectionEnv
import pandas as pd

# Load your questions data
questions_df = pd.read_csv('path/to/your/questions.csv')

# Create the environment
env = QuestionSelectionEnv(questions_df=questions_df, max_steps=100)

# Initialize the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("trained_question_selector")