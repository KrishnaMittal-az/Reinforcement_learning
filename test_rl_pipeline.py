"""
Test Script for RL Pipeline with Working DKT Model
Tests complete pipeline: DKT model → RL Environment → PPO Agent
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Add colored output for better visibility
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'
BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*70}")
    print(message)
    print(f"{'='*70}{END}\n")

def test_dkt_model(model_path='models/dkt_model_working.keras'):
    """Test if DKT model works and show its predictions"""
    print_header("1. TESTING DKT MODEL")
    
    try:
        # Load model
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"{GREEN}✓ Model loaded successfully{END}")
        
        # Print model summary
        print("\nModel Architecture:")
        model.summary()
        
        # Test prediction
        print("\nTesting prediction with dummy data...")
        dummy_input = np.random.randint(0, 2, size=(1, 10, 100))  # Batch=1, Seq=10, Features=100
        prediction = model.predict(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction[0][0]:.4f}")
        
        return model
        
    except Exception as e:
        print(f"{RED}❌ Error testing model: {str(e)}{END}")
        raise

def test_environment(model):
    """Test if RL environment works with the model"""
    print_header("2. TESTING RL ENVIRONMENT")
    
    try:
        from environment.question_selection_env_main import QuestionSelectionEnv
        
        # Load questions
        print("Loading question database...")
        questions_df = pd.read_csv('data/questions.csv')
        print(f"{GREEN}✓ Loaded {len(questions_df)} questions{END}")
        
        # Initialize environment
        print("\nInitializing environment...")
        env = QuestionSelectionEnv(
            questions_df,
            max_steps=10,
            action_types=['skill', 'type'],
            lstm_model_path='models/dkt_model_working.keras'  # Use path to our working model
        )
        print(f"{GREEN}✓ Environment initialized{END}")
        
        # Test reset
        print("\nTesting environment reset...")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        return env
        
    except Exception as e:
        print(f"{RED}❌ Error testing environment: {str(e)}{END}")
        raise

def test_ppo_agent(env):
    """Test training a PPO agent"""
    print_header("3. TESTING PPO AGENT")
    
    try:
        # Create PPO agent
        print("Creating PPO agent...")
        ppo = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./trainings/PPO_test/"
        )
        print(f"{GREEN}✓ PPO agent created{END}")
        
        # Train for a few steps
        print("\nTraining for 1000 timesteps...")
        ppo.learn(total_timesteps=1000)
        print(f"{GREEN}✓ Training complete{END}")
        
        return ppo
        
    except Exception as e:
        print(f"{RED}❌ Error testing PPO: {str(e)}{END}")
        raise

def test_interaction(env, ppo):
    """Test full interaction loop"""
    print_header("4. TESTING INTERACTION LOOP")
    
    try:
        print("Running 5 episodes with trained agent...")
        
        for episode in range(5):
            print(f"\n{BLUE}Episode {episode+1}{END}")
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                # Agent selects action
                action, _ = ppo.predict(obs)
                
                # Environment step
                obs, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Print step info
                print(f"Step {steps}: Action={action}, Reward={reward:.2f}")
                
                if 'question' in info:
                    print(f"Question: {info['question']}")
                    print(f"Skill: {info['skill']}")
                    print(f"Difficulty: {info['difficulty']}")
                
                if done:
                    print(f"\nEpisode finished after {steps} steps")
                    print(f"Total reward: {total_reward:.2f}")
                    break
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ Error in interaction: {str(e)}{END}")
        raise

def main():
    """Run complete pipeline test"""
    print_header("TESTING COMPLETE RL PIPELINE")
    
    try:
        # 1. Test DKT Model
        model = test_dkt_model()
        
        # 2. Test Environment
        env = test_environment(model)
        
        # 3. Test PPO Agent
        ppo = test_ppo_agent(env)
        
        # 4. Test Interaction
        success = test_interaction(env, ppo)
        
        if success:
            print(f"\n{GREEN}{BOLD}🎉 ALL TESTS PASSED!{END}")
            print("The RL pipeline is working correctly.")
            return 0
            
    except Exception as e:
        print(f"\n{RED}{BOLD}❌ TEST FAILED{END}")
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())