"""
Demo: Run a Personalized Learning Session

This script demonstrates how the RL system works in practice.
It simulates a student learning session with the trained teacher agent.
"""

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment.question_selection_env_main import QuestionSelectionEnv
import json
import os

def run_demo_session(num_questions=20, model_path=None, save_history=True):
    """
    Run a demo learning session
    
    Args:
        num_questions: Number of questions to ask (default: 20)
        model_path: Path to trained PPO model (optional, uses random if None)
        save_history: Whether to save session history to JSON
    """
    
    print("="*70)
    print("  PERSONALIZED LEARNING SYSTEM - DEMO SESSION")
    print("="*70)
    print()
    
    # Load question bank
    print("📚 Loading question bank...")
    questions_df = pd.read_csv("./data/questions.csv")
    print(f"   ✓ Loaded {len(questions_df)} questions")
    print()
    
    # Create environment
    print("🏗️  Initializing learning environment...")
    env = QuestionSelectionEnv(
        questions_df,
        max_steps=num_questions,
        action_types=['skill', 'type'],  # Teacher controls skill and question type
        w_improvement=100,
        w_answerability=50,
        w_coverage=0.5
    )
    print("   ✓ Environment ready")
    print(f"   • Tracking {env.num_skills} skills")
    print(f"   • {env.num_question_types} question types available")
    print()
    
    # Load teacher model (or use random)
    if model_path and os.path.exists(model_path):
        print(f"🤖 Loading trained teacher agent from {model_path}...")
        teacher = PPO.load(model_path)
        print("   ✓ Teacher loaded (using learned strategy)")
    else:
        print("🎲 No trained model found - using random strategy")
        teacher = None
    print()
    
    # Start session
    print("🎓 Starting learning session...")
    print("-"*70)
    print()
    
    obs = env.reset()
    total_reward = 0
    skill_improvements = {}
    
    for step in range(num_questions):
        # Teacher selects action
        if teacher:
            action, _ = teacher.predict(obs, deterministic=True)
        else:
            # Random action
            action = env.action_space.sample()
        
        # Execute action
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        total_reward += reward
        
        # Track skill improvement
        skill = info['skill']
        if skill not in skill_improvements:
            skill_improvements[skill] = []
        skill_improvements[skill].append(info['predicted_correctness_for_skill'])
        
        # Display progress
        print(f"📝 Question {step + 1}/{num_questions}")
        print(f"   Question: {info['question'][:70]}...")
        print(f"   Skill: {skill}")
        print(f"   Type: {info['question_type']}")
        print(f"   Student Mastery: {info['predicted_correctness_for_skill']:.1%}")
        print(f"   Reward Breakdown:")
        print(f"      • Improvement: {info['improvement']:+.2f}")
        print(f"      • Answerability: {info['answerability']:+.2f}")
        print(f"      • Weak Skill Coverage: {info['coverage']:+.2f}")
        print(f"   Total Reward: {reward:+.2f}")
        print()
        
        if done:
            break
    
    # Session summary
    print("="*70)
    print("  SESSION SUMMARY")
    print("="*70)
    print()
    print(f"📊 Total Questions: {step + 1}")
    print(f"🎯 Total Reward: {total_reward:.2f}")
    print(f"📈 Average Reward per Question: {total_reward / (step + 1):.2f}")
    print()
    
    # Analyze skill coverage
    print("🎯 Skills Practiced:")
    for skill, performances in skill_improvements.items():
        initial = performances[0]
        final = performances[-1]
        improvement = final - initial
        print(f"   • {skill[:40]:40} | Start: {initial:.1%} → End: {final:.1%} ({improvement:+.1%})")
    print()
    
    # Calculate overall improvement
    all_initial = [perfs[0] for perfs in skill_improvements.values()]
    all_final = [perfs[-1] for perfs in skill_improvements.values()]
    avg_improvement = np.mean(all_final) - np.mean(all_initial)
    
    print(f"📊 Overall Learning Metrics:")
    print(f"   • Skills Addressed: {len(skill_improvements)}")
    print(f"   • Average Initial Mastery: {np.mean(all_initial):.1%}")
    print(f"   • Average Final Mastery: {np.mean(all_final):.1%}")
    print(f"   • Total Improvement: {avg_improvement:+.1%}")
    print()
    
    # Top 5 weakest skills
    final_performance = {k: v[-1] for k, v in skill_improvements.items()}
    weakest = sorted(final_performance.items(), key=lambda x: x[1])[:5]
    print("⚠️  Top 5 Skills Needing Practice:")
    for i, (skill, perf) in enumerate(weakest, 1):
        print(f"   {i}. {skill[:50]:50} | Mastery: {perf:.1%}")
    print()
    
    # Save history
    if save_history:
        history_path = "demo_session_history.json"
        env.save_history_json(history_path)
        print(f"💾 Session history saved to: {history_path}")
        print()
    
    print("="*70)
    print("  SESSION COMPLETE")
    print("="*70)
    
    return env.history


def analyze_training_history(history_path="trainings/history/history_80k_skill&QT.json"):
    """
    Analyze a previously saved training history
    
    Args:
        history_path: Path to history JSON file
    """
    print("="*70)
    print("  TRAINING HISTORY ANALYSIS")
    print("="*70)
    print()
    
    if not os.path.exists(history_path):
        print(f"❌ History file not found: {history_path}")
        return
    
    print(f"📂 Loading history from: {history_path}")
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"   ✓ Loaded {len(history)} interactions")
    print()
    
    # Extract metrics
    rewards = [h['reward'] for h in history]
    improvements = [h['improvement'] for h in history]
    answerabilities = [h['answerability'] for h in history]
    coverages = [h['coverage'] for h in history]
    skills = [h['skill'] for h in history]
    
    # Statistics
    print("📊 Reward Statistics:")
    print(f"   • Total Reward: {sum(rewards):.2f}")
    print(f"   • Average Reward: {np.mean(rewards):.2f}")
    print(f"   • Max Reward: {np.max(rewards):.2f}")
    print(f"   • Min Reward: {np.min(rewards):.2f}")
    print()
    
    print("📈 Component Analysis:")
    print(f"   • Average Improvement: {np.mean(improvements):.2f}")
    print(f"   • Average Answerability: {np.mean(answerabilities):.2f}")
    print(f"   • Average Coverage: {np.mean(coverages):.2f}")
    print()
    
    print("🎯 Skill Coverage:")
    unique_skills = set(skills)
    print(f"   • Total Unique Skills: {len(unique_skills)}")
    
    # Most practiced skills
    from collections import Counter
    skill_counts = Counter(skills)
    print(f"   • Most Practiced Skills:")
    for skill, count in skill_counts.most_common(10):
        print(f"      - {skill[:40]:40} | {count} times ({count/len(history)*100:.1f}%)")
    print()
    
    # Improvement over time (in chunks)
    chunk_size = len(history) // 10
    print("📊 Learning Progress (10 segments):")
    for i in range(10):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_reward = np.mean(rewards[start:end])
        print(f"   Segment {i+1:2d} (steps {start:5d}-{end:5d}): Avg Reward = {chunk_reward:+.2f}")
    print()
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "demo":
            # Run demo session
            num_q = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            model = sys.argv[3] if len(sys.argv) > 3 else None
            run_demo_session(num_questions=num_q, model_path=model)
            
        elif command == "analyze":
            # Analyze history
            path = sys.argv[2] if len(sys.argv) > 2 else "trainings/history/history_80k_skill&QT.json"
            analyze_training_history(path)
            
        else:
            print("Usage:")
            print("  python demo_session.py demo [num_questions] [model_path]")
            print("  python demo_session.py analyze [history_path]")
    
    else:
        # Default: run small demo
        print("Running default demo (20 questions, random teacher)...")
        print()
        run_demo_session(num_questions=20, model_path=None)
        print()
        print("To run with trained teacher:")
        print("  python demo_session.py demo 50 models/teacher/ppo_teacher_agent.zip")
        print()
        print("To analyze training history:")
        print("  python demo_session.py analyze trainings/history/history_80k_skill&QT.json")
