# RL Project Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Run a Demo (No Training Required)

```bash
# Navigate to the RL project
cd rl-proj/personalized_learning_rl

# Install dependencies (one-time setup)
pip install -r requirements.txt

# Run demo with pre-trained models
python demo_session.py demo 20
```

This will:
- Load pre-trained student model (LSTM)
- Run 20-question learning session
- Show real-time question selection
- Display learning progress

**Expected Output**:
```
📚 Loading question bank...
   ✓ Loaded 1000 questions

🤖 Loading trained teacher agent...
   ✓ Teacher loaded (using learned strategy)

📝 Question 1/20
   Question: What is the absolute value of -15?
   Skill: Absolute Value
   Type: fill_in_one
   Student Mastery: 12.3%
   Reward: +2.45

...
```

---

## 📖 Understanding the Demo Output

### Question Information
```
Question: Add: 234 + 567                    ← The actual question
Skill: Addition Whole Numbers               ← Math skill being tested
Type: Multiple Choice                       ← Question format
Student Mastery: 45.2%                      ← Predicted success rate
```

### Reward Breakdown
```
Reward Breakdown:
  • Improvement: +1.23     ← Learning gain (weighted ×100)
  • Answerability: +22.5   ← Question difficulty match (×50)
  • Weak Skill Coverage: +0.5  ← Targets weak areas (×0.5)
Total Reward: +24.23
```

**What the numbers mean**:
- **Improvement** (-10 to +10): Change in average skill mastery
  - Positive = Student learning
  - Negative = Performance drop (shouldn't happen often)
  
- **Answerability** (0 to 50): How well-matched the question is
  - 0-15: Too hard (student will struggle)
  - 15-35: Perfect (challenging but doable)
  - 35-50: Too easy (won't learn much)
  
- **Coverage** (0 or 0.5): Whether it targets weak skills
  - 0.5: Focuses on bottom 40% of skills ✓
  - 0: Not a priority skill

---

## 🎯 Option 2: Run a Complete Training Session

### Step 1: Prepare Data (Optional - Data Included)

```bash
# If you want to re-process the raw data
jupyter notebook data_cleaning.ipynb
# Run all cells → creates cleaned_df.csv
```

### Step 2: Train Student Model (Optional - Pre-trained Available)

```bash
# Open student training notebook
jupyter notebook agents/student_lstm_agent_kt-skill+qt.ipynb

# Run all cells sequentially
# This trains the LSTM model that simulates student learning
# Time: ~2-3 hours on CPU, ~30 min on GPU
```

**What it does**:
1. Loads 325,000 real student interactions
2. Creates sequences of student history
3. Trains LSTM to predict performance
4. Saves model to `models/dkt_model_pretrained_seq50.keras`

**Key Metrics to Watch**:
- Validation AUC: Should reach 0.85+
- Validation Accuracy: Should reach 75%+

### Step 3: Train Teacher Agent

```bash
# Train the RL teacher
python agents/teacher_agent.py
```

**What happens**:
```
Starting training...
Step 1 | Question: Add: 234 + 567 | Skill: Addition | Reward: +12.3
Step 2 | Question: Solve: x + 5 = 12 | Skill: Algebra | Reward: +15.7
...
Step 250 | Total Episode Reward: 1234.5
----------------------------------------
Episode 1/800 Complete | Reward: 1234.5
Episode 2/800 Complete | Reward: 1456.2
...
```

**Training Time**:
- 200,000 timesteps = ~800 episodes
- CPU: 6-8 hours
- GPU: 2-3 hours

**Monitor Training**:
```bash
# In another terminal
tensorboard --logdir=./trainings
# Open http://localhost:6006
```

---

## 📊 Analyzing Results

### After Training - View Logs

```bash
# Analyze saved history
python demo_session.py analyze trainings/history/history_80k_skill&QT.json
```

**Output**:
```
📊 Reward Statistics:
   • Total Reward: 45,678.90
   • Average Reward: 15.23
   • Max Reward: 48.90
   • Min Reward: -5.23

📈 Component Analysis:
   • Average Improvement: 0.012
   • Average Answerability: 22.5
   • Average Coverage: 0.25

🎯 Skill Coverage:
   • Total Unique Skills: 142/170
   • Most Practiced Skills:
      - Algebraic Solving                      | 234 times (7.8%)
      - Addition and Subtraction Fractions     | 189 times (6.3%)
      ...
```

### Visualize Training Progress

```bash
jupyter notebook plotting.ipynb
```

**Charts Available**:
1. **Reward over Time**: Shows learning curve
2. **Skill Coverage**: Which skills were practiced
3. **Question Type Distribution**: Mix of question formats
4. **Improvement Trends**: Student mastery growth

---

## 🔧 Customization Guide

### Adjust Reward Weights

**File**: `agents/teacher_agent.py`

```python
env = QuestionSelectionEnv(
    questions_df,
    w_improvement=100,      # ← Increase to prioritize learning
    w_answerability=50,     # ← Increase for better difficulty matching
    w_coverage=0.5,         # ← Increase to focus more on weak skills
    weak_skills_threshold=0.4  # ← Lower = target more skills (0.3-0.5)
)
```

**Experiment Ideas**:
```python
# Focus heavily on weak skills
env = QuestionSelectionEnv(..., w_coverage=5.0, weak_skills_threshold=0.3)

# Prioritize perfect difficulty matching
env = QuestionSelectionEnv(..., w_answerability=80, w_improvement=50)

# Maximum learning gain
env = QuestionSelectionEnv(..., w_improvement=150, w_coverage=0.0)
```

### Change Training Duration

**File**: `agents/teacher_agent.py`

```python
# Quick test (1 hour)
model.learn(total_timesteps=50000)

# Standard (6-8 hours) - DEFAULT
model.learn(total_timesteps=200000)

# Extended training (12+ hours)
model.learn(total_timesteps=500000)
```

### Modify Action Space

**File**: `agents/teacher_agent.py`

```python
# Teacher controls skill only (random question type)
env = QuestionSelectionEnv(
    questions_df,
    action_types=['skill']
)

# Teacher controls both skill and question type - DEFAULT
env = QuestionSelectionEnv(
    questions_df,
    action_types=['skill', 'type']
)

# Baseline: Random selection
env = QuestionSelectionEnv(
    questions_df,
    action_types=[]
)
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory

**Error**: `OOM when allocating tensor`

**Solution**:
```python
# Reduce sequence length
env = QuestionSelectionEnv(..., max_seq_len=30)  # Default: 50

# Reduce batch size in PPO
model = PPO(..., batch_size=128, n_steps=1024)  # Default: 256, 2048
```

### Issue 2: Reward Not Increasing

**Symptoms**: Episode reward stays flat around 500-800

**Possible Causes**:
1. Student model accuracy too low
2. Learning rate too high/low
3. Reward weights not balanced

**Solutions**:
```python
# 1. Check student model performance
# Open student_lstm_agent_kt-skill+qt.ipynb
# Look for: Validation AUC > 0.8

# 2. Adjust learning rate
model = PPO(..., learning_rate=0.0002)  # Default: 0.0004

# 3. Balance rewards
env = QuestionSelectionEnv(
    w_improvement=150,  # Increase if too low
    w_answerability=30,  # Decrease if too high
)
```

### Issue 3: Training Too Slow

**Solution 1 - Use Vectorized Environments**:
```python
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env(lambda: env, n_envs=4)  # 4 parallel envs
model = PPO("MlpPolicy", vec_env, ...)
```

**Solution 2 - Reduce Steps**:
```python
env = QuestionSelectionEnv(..., max_steps=150)  # Default: 250
```

### Issue 4: Import Errors

**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**:
```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade

# Or install individually
pip install transformers sentence-transformers
```

### Issue 5: Model File Not Found

**Error**: `FileNotFoundError: models/dkt_model_pretrained_seq50.keras`

**Solution**:
```bash
# Train student model first
jupyter notebook agents/student_lstm_agent_kt-skill+qt.ipynb
# Run all cells to generate the model

# OR change model path in teacher_agent.py
lstm_model_path="./models/dkt_model_pretrained.keras"  # Different model
```

---

## 📈 Interpreting Training Curves

### TensorBoard Metrics

```bash
tensorboard --logdir=./trainings
```

**Key Metrics**:

1. **rollout/ep_rew_mean** (Episode Reward)
   - **Good**: Steady increase from 500 → 2000+
   - **Bad**: Flat or decreasing
   - **Fix**: Adjust learning rate, check student model

2. **train/value_loss** (Value Function Loss)
   - **Good**: Decreases over time
   - **Bad**: Oscillates wildly
   - **Fix**: Lower learning rate, increase n_steps

3. **train/policy_loss** (Policy Loss)
   - **Good**: Stabilizes around -0.01 to -0.001
   - **Bad**: Doesn't converge
   - **Fix**: Check reward scaling, adjust clip_range

4. **train/entropy_loss** (Exploration)
   - **Good**: Gradually decreases (from 1.0 → 0.3)
   - **Bad**: Drops to zero immediately
   - **Fix**: Increase ent_coef

---

## 🎓 Example Workflow

### Scenario: Train a Teacher for Algebra-Heavy Curriculum

```python
# Step 1: Filter questions to algebra
questions_df = pd.read_csv("data/questions.csv")
algebra_questions = questions_df[
    questions_df['skill'].str.contains('Algebra|Equation|Solving')
]

# Step 2: Create specialized environment
env = QuestionSelectionEnv(
    algebra_questions,
    max_steps=200,
    w_improvement=150,  # Heavy focus on learning
    weak_skills_threshold=0.3  # Target bottom 30%
)

# Step 3: Train longer
model = PPO("MlpPolicy", env, learning_rate=0.0003)
model.learn(total_timesteps=300000)  # Extended training

# Step 4: Save specialized model
model.save("models/teacher/algebra_specialist")
```

---

## 🌟 Success Metrics

Your RL system is working well when:

✅ **Episode Reward > 1500** (after 200k steps)
✅ **Skill Coverage > 60%** (addresses most weak skills)
✅ **Avg Answerability: 20-30** (good difficulty match)
✅ **Student Mastery Increases** over session
✅ **Reward Curve Trending Upward**

---

## 🚀 Next Steps

1. **Experiment with Hyperparameters**
   - Try different reward weights
   - Adjust learning rates
   - Vary sequence lengths

2. **Add New Features**
   - Time constraints (timed questions)
   - Student fatigue modeling
   - Curriculum dependencies

3. **Deploy to Production**
   - Integrate with web app
   - Add real student feedback
   - Track long-term outcomes

4. **Advanced RL Algorithms**
   - Try SAC (Soft Actor-Critic)
   - Experiment with A2C
   - Test DQN for discrete actions

---

## 📚 File Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `demo_session.py` | Test trained model | After training |
| `teacher_agent.py` | Train RL teacher | Main training script |
| `student_lstm_agent_kt-skill+qt.ipynb` | Train student model | Initial setup |
| `data_cleaning.ipynb` | Process raw data | If you have new data |
| `plotting.ipynb` | Visualize results | Analyze training |
| `question_selection_env.py` | RL environment | Modify for custom logic |

---

## 💡 Pro Tips

1. **Start Small**: Run 50k timesteps first to test setup
2. **Monitor Early**: Open TensorBoard from the start
3. **Save Checkpoints**: Model is auto-saved every update
4. **Compare Baselines**: Train with `action_types=[]` (random) as baseline
5. **Track Everything**: Environment saves detailed history automatically

---

**Questions?** Check the full guide: `RL_PROJECT_EXPLAINED.md`

---

*Last Updated: October 2025*
