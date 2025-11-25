# Study Roadmap: Reinforcement Learning Based Personalized Education Tutor

This roadmap is designed to help you understand, use, and explain your RL-based personalized education tutor project.

## 1. Project Overview
**Goal:** To create an intelligent tutoring system that adapts to each student's learning pace and knowledge gaps.
**Core Concept:**
*   **Student Model (DKT):** A "Brain" that estimates how well a student knows different skills based on their past answers.
*   **Teacher Agent (RL):** An "AI Tutor" that looks at the student's "Brain" state and decides which question to ask next to maximize learning.
*   **Environment:** A simulation where the Student and Teacher interact.

## 2. Prerequisites
Before diving deep, ensure you have a basic understanding of:
*   **Python:** Intermediate level (classes, data manipulation with Pandas).
*   **Machine Learning:** Basic concepts of Neural Networks (LSTM).
*   **Reinforcement Learning:**
    *   **Agent:** The learner/decision maker (The Teacher).
    *   **Environment:** The world the agent interacts with (The Quiz Session).
    *   **State:** The current situation (Student's knowledge profile).
    *   **Action:** What the agent does (Select a question on a specific skill).
    *   **Reward:** Feedback on how good the action was (Did the student learn? Did they answer correctly?).
    *   **PPO (Proximal Policy Optimization):** The specific algorithm used to train the Teacher.

## 3. Study Roadmap (Step-by-Step)

### Step 1: The Data (Foundation)
*   **Files:** `data/questions.csv`, `curriculum_skills.json` (or `sorted_skills.json`)
*   **Task:** Open `data/questions.csv`.
*   **What to look for:**
    *   Columns: `question_text`, `skill`, `difficulty`, `question_type`.
    *   This is the "Knowledge Base" the teacher draws from.
*   **Goal:** Understand what raw material the system works with.

### Step 2: The Student Model (The "Brain")
*   **Files:** `models/dkt_model_working.keras` (Binary), `environment/question_selection_env_main.py` (Usage)
*   **Concept:** Deep Knowledge Tracing (DKT). It uses an LSTM (Long Short-Term Memory) network.
*   **How it works:**
    *   Input: Sequence of (Skill, Correct/Incorrect).
    *   Output: Probability of answering correctly for *all* skills in the future.
*   **Study:** Look at `environment/question_selection_env_main.py` around line 100-200 where the model is loaded and used to predict `student_performance`.

### Step 3: The Environment (The "Classroom")
*   **Files:** `environment/question_selection_env_main.py`
*   **Task:** Read the `QuestionSelectionEnv` class.
*   **Key Methods:**
    *   `__init__`: Sets up the student model and question bank.
    *   `step(action)`: The core loop.
        1.  Teacher picks a skill (`action`).
        2.  Environment picks a specific question for that skill.
        3.  Student answers (simulated or real).
        4.  Student Model updates its belief (`student_performance`).
        5.  **Reward** is calculated (Improvement in mastery + Correctness).
    *   `reset()`: Starts a new student session.

### Step 4: The Teacher Agent (The "AI")
*   **Files:** `agents/teacher_agent.py`
*   **Concept:** PPO (Proximal Policy Optimization) from `stable_baselines3`.
*   **Task:**
    *   See how `make_vec_env` creates the environment.
    *   See how `PPO(...)` initializes the agent.
    *   See `model.learn(...)` which runs the training loop (Trial & Error).
    *   See `model.save(...)` which saves the trained "brain" of the teacher.

### Step 5: The Web Application (The Interface)
*   **Files:** `app.py`, `templates/`
*   **Task:**
    *   See how `Flask` is used to create a web server.
    *   Look at `_select_action`: It tries to use the trained PPO agent. If not found, it uses a simple rule (ask the weakest skill).
    *   See how it renders `quiz.html`.

## 4. How to Use It

### Setup
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to fix `requirements.txt` if versions clash. See `requirements_fixed.txt` if available)*

### Training the Teacher (Optional but Recommended)
If you want to make the AI smarter:
1.  Run the training script:
    ```bash
    python agents/teacher_agent.py
    ```
    *(Note: Ensure `agents/teacher_agent.py` imports from `environment.question_selection_env_main` instead of `environment.question_selection_env` if the latter is empty)*
2.  This will create `models/teacher/ppo_teacher_agent.zip`.

### Running the Quiz App
1.  Start the server:
    ```bash
    python app.py
    ```
2.  Open your browser at `http://127.0.0.1:5000`.
3.  Take the quiz! The system will adapt to you.

## 5. How to Explain It (The "Elevator Pitch")

**"I built an AI Tutor that doesn't just give you random questions. It learns how you learn."**

*   **The Problem:** Standard quizzes are "one size fits all". They are either too hard (frustrating) or too easy (boring).
*   **The Solution:** My project uses **Reinforcement Learning**.
    *   Imagine a human tutor watching you solve problems. They notice, "Oh, you're good at Algebra but struggling with Geometry."
    *   My **Student Model (DKT)** does exactly that mathematically—it tracks your mastery of every skill in real-time.
    *   My **Teacher Agent (RL)** uses that information to pick the *perfect* next question to maximize your learning speed.
*   **The Tech:** It uses Python, TensorFlow (for the student brain), Stable Baselines3 (for the teacher brain), and Flask (for the web app).

## 6. Troubleshooting & Notes
*   **Empty File Warning:** `environment/question_selection_env.py` might be empty. The real code seems to be in `environment/question_selection_env_main.py`. If you run scripts that fail importing the environment, check which file they are trying to import.
*   **Model Paths:** Ensure `models/dkt_model_working.keras` exists. The system needs this to simulate the student.
