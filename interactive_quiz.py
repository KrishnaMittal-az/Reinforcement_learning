"""
Simple Interactive Adaptive Quiz - Terminal Based
Users answer questions and the system adapts based on their performance
"""

import pandas as pd
import numpy as np
from environment.question_selection_env_main import QuestionSelectionEnv
import os

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_welcome():
    """Display welcome message"""
    clear_screen()
    print("="*70)
    print("        ADAPTIVE MATH QUIZ - Personalized Learning System")
    print("="*70)
    print()
    print("This quiz adapts to YOUR learning level!")
    print("• Answer questions to the best of your ability")
    print("• The system will adjust difficulty based on your performance")
    print("• Track your progress across different math skills")
    print()
    input("Press ENTER to start...")

def display_question(question_num, total_questions, question_info):
    """Display a single question"""
    clear_screen()
    print("="*70)
    print(f"Question {question_num}/{total_questions}")
    print("="*70)
    print()
    print(f"📚 Topic: {question_info['skill']}")
    print(f"📊 Difficulty: {question_info['difficulty']}")
    print()
    print(f"❓ {question_info['question_text']}")
    print()
    
    # Display choices if multiple choice
    if question_info['question_type'] == 'multiple_choice' and question_info['choices']:
        choices = question_info['choices'].split('|')
        print("Choose from:")
        for i, choice in enumerate(choices, 1):
            print(f"   {i}. {choice}")
        print()

def get_user_answer(question_info):
    """Get answer from user"""
    if question_info['question_type'] == 'multiple_choice' and question_info['choices']:
        choices = question_info['choices'].split('|')
        while True:
            try:
                choice_num = int(input("Your choice (enter number): "))
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
    else:
        return input("Your answer: ")

def check_answer(user_answer, correct_answer):
    """Check if answer is correct"""
    # Normalize answers for comparison
    user = str(user_answer).strip().lower().replace(" ", "")
    correct = str(correct_answer).strip().lower().replace(" ", "")
    
    # Handle various formats
    return user == correct or user in correct or correct in user

def display_feedback(is_correct, correct_answer, skill_mastery):
    """Display feedback to user"""
    print()
    print("-"*70)
    if is_correct:
        print("✅ CORRECT! Well done!")
    else:
        print(f"❌ Not quite. The correct answer was: {correct_answer}")
    
    print()
    print(f"Your current mastery of '{skill_mastery['skill']}': {skill_mastery['mastery']:.0%}")
    print("-"*70)
    print()
    input("Press ENTER to continue...")

def display_progress(stats):
    """Display current progress"""
    print()
    print("📊 Your Progress So Far:")
    print(f"   Correct: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.0f}%)")
    print(f"   Topics covered: {len(stats['skills_practiced'])}")
    print()

def display_final_summary(stats, skill_performance):
    """Display final summary"""
    clear_screen()
    print("="*70)
    print("                    QUIZ COMPLETE!")
    print("="*70)
    print()
    print(f"📊 Final Score: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.0f}%)")
    print()
    
    print("🎯 Skills You Practiced:")
    for skill, performances in skill_performance.items():
        final_mastery = performances[-1]
        emoji = "🟢" if final_mastery > 0.7 else "🟡" if final_mastery > 0.4 else "🔴"
        print(f"   {emoji} {skill[:45]:45} | Mastery: {final_mastery:.0%}")
    print()
    
    # Overall improvement
    all_final = [perfs[-1] for perfs in skill_performance.values()]
    avg_mastery = np.mean(all_final)
    print(f"📈 Overall Average Mastery: {avg_mastery:.0%}")
    print()
    
    # Recommendations
    weak_skills = [(skill, perf[-1]) for skill, perf in skill_performance.items() if perf[-1] < 0.5]
    if weak_skills:
        print("💡 Recommended Topics to Practice:")
        for skill, mastery in sorted(weak_skills, key=lambda x: x[1])[:5]:
            print(f"   • {skill} (Current: {mastery:.0%})")
    else:
        print("🌟 Excellent work! You're doing great across all topics!")
    
    print()
    print("="*70)

def run_interactive_quiz(num_questions=10):
    """
    Main function to run interactive quiz
    
    Args:
        num_questions: Number of questions to ask (default: 10)
    """
    # Welcome
    display_welcome()
    
    # Load questions and setup environment
    print("Loading quiz system...")
    questions_df = pd.read_csv("./data/questions.csv")
    
    env = QuestionSelectionEnv(
        questions_df,
        max_steps=num_questions,
        action_types=['skill', 'type'],
        w_improvement=100,
        w_answerability=50,
        w_coverage=0.5
    )
    
    # Initialize tracking
    obs = env.reset()
    stats = {
        'correct': 0,
        'total': 0,
        'skills_practiced': set()
    }
    skill_performance = {}
    
    print("✓ Ready!")
    print()
    input("Press ENTER to begin...")
    
    # Quiz loop
    for question_num in range(1, num_questions + 1):
        # Get next question (environment selects adaptively)
        action = env.action_space.sample()  # You can replace with trained model
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        question_info = info
        skill = info['skill']
        
        # Track skill performance
        if skill not in skill_performance:
            skill_performance[skill] = []
        skill_performance[skill].append(info['predicted_correctness_for_skill'])
        
        # Display question
        display_question(question_num, num_questions, question_info)
        
        # Get user answer
        user_answer = get_user_answer(question_info)
        
        # Check answer
        is_correct = check_answer(user_answer, question_info['answer'])
        
        # Update stats
        stats['total'] += 1
        if is_correct:
            stats['correct'] += 1
        stats['skills_practiced'].add(skill)
        
        # Show feedback
        display_feedback(
            is_correct, 
            question_info['answer'],
            {
                'skill': skill,
                'mastery': info['predicted_correctness_for_skill']
            }
        )
        
        # Show progress every 3 questions
        if question_num % 3 == 0 and question_num < num_questions:
            display_progress(stats)
            input("Press ENTER to continue...")
        
        if done:
            break
    
    # Final summary
    display_final_summary(stats, skill_performance)
    
    # Save history
    env.save_history_json("quiz_session_history.json")
    print("\n💾 Your session has been saved to: quiz_session_history.json")
    print()

if __name__ == "__main__":
    import sys
    
    # Get number of questions from command line or use default
    if len(sys.argv) > 1:
        try:
            num_q = int(sys.argv[1])
        except ValueError:
            num_q = 10
            print(f"Invalid number, using default: {num_q} questions")
    else:
        num_q = 10
    
    print(f"Starting quiz with {num_q} questions...")
    print()
    
    run_interactive_quiz(num_questions=num_q)
