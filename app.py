"""Minimal demo web app for the personalized learning RL pipeline.

Run with:
    python app.py
then open http://127.0.0.1:5000/ in a browser.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from stable_baselines3 import PPO

from environment.question_selection_env_main import QuestionSelectionEnv

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "questions.csv"
MODEL_PATH = BASE_DIR / "models" / "teacher" / "ppo_policy.zip"
MAX_QUESTIONS = 10

app = Flask(__name__)
app.secret_key = os.environ.get("QUIZ_APP_SECRET", "demo-secret")

# Load questions once
QUESTIONS_DF = pd.read_csv(DATA_PATH)

# Try loading a trained PPO policy (optional)
AGENT: Optional[PPO] = None
if MODEL_PATH.exists():
    try:
        AGENT = PPO.load(MODEL_PATH)
        print(f"Loaded PPO policy from {MODEL_PATH}")
    except Exception as exc:  # pragma: no cover - informational
        print(f"Warning: failed to load PPO policy: {exc}")
        AGENT = None


@dataclass
class QuizSession:
    env: QuestionSelectionEnv
    obs: np.ndarray
    steps: int = 0
    total_reward: float = 0.0
    done: bool = False
    last_info: Optional[Dict] = None
    current_question: Optional[Dict] = None
    current_reward: float = 0.0
    history: List[Dict] = field(default_factory=list)


SESSIONS: Dict[str, QuizSession] = {}


def _select_action(session: QuizSession) -> np.ndarray:
    """Choose the next action using the PPO policy if available, else heuristic."""
    if AGENT is not None:
        action, _ = AGENT.predict(session.obs, deterministic=True)
        return np.array(action, dtype=np.int64)

    # Simple heuristic: target the weakest skill seen so far
    weakest_skill = int(np.argmin(session.env.student_performance))
    question_type = session.env.action_space.sample()[1]
    return np.array([weakest_skill, question_type], dtype=np.int64)


def _format_question(info: Dict, reward: float) -> Dict:
    metadata = info.get("question_metadata", {})
    raw_choices = metadata.get("choices")
    choices: Optional[List[str]] = None
    if isinstance(raw_choices, str) and raw_choices.strip():
        choices = [item.strip() for item in raw_choices.split("|") if item.strip()]

    return {
        "prompt": info.get("question", ""),
        "skill": info.get("skill", "Unknown"),
        "difficulty": info.get("difficulty", "unknown"),
        "question_type": info.get("question_type", ""),
        "choices": choices,
        "answer": metadata.get("answer", ""),
        "reward": reward,
    }
def _get_next_question(session: QuizSession) -> Optional[Dict]:
    """Get the next adaptive question without repeating any previously asked ones."""
    if session.done:
        return None

    # --- Filter out already-used questions ---
    used_prompts = {q["prompt"] for q in session.history if "prompt" in q}
    available_questions = session.env.questions_df[
        ~session.env.questions_df["question_text"].isin(used_prompts)
    ]

    # If we've run out of unique questions, end session early
    if available_questions.empty:
        session.done = True
        return None

    # Update environment question pool temporarily
    session.env.questions_df = available_questions

    # --- RL Step ---
    action = _select_action(session)
    obs, reward, terminated, truncated, info = session.env.step(action)

    # --- Update session state ---
    session.obs = obs
    session.total_reward += float(reward)
    session.steps += 1
    session.last_info = info
    session.current_reward = float(reward)
    session.done = bool(terminated or truncated or session.steps >= MAX_QUESTIONS)
    session.current_question = _format_question(info, reward)

    return session.current_question


# def _get_next_question(session: QuizSession) -> Optional[Dict]:
#     if session.done:
#         return None

#     action = _select_action(session)
#     obs, reward, terminated, truncated, info = session.env.step(action)
#     session.obs = obs
#     session.total_reward += float(reward)
#     session.steps += 1
#     session.last_info = info
#     session.current_reward = float(reward)
#     session.done = bool(terminated or truncated or session.steps >= MAX_QUESTIONS)
#     session.current_question = _format_question(info, reward)
#     return session.current_question


def _evaluate_answer(expected: str, user_answer: str) -> bool:
    if expected is None:
        return False
    return str(user_answer).strip().lower() == str(expected).strip().lower()


def _start_session() -> str:
    env = QuestionSelectionEnv(
        questions_df=QUESTIONS_DF.copy(),
        max_steps=MAX_QUESTIONS,
        action_types=['skill', 'type'],
        lstm_model_path=str(BASE_DIR / "models" / "dkt_model_working.keras"),
    )
    obs, _ = env.reset()
    session = QuizSession(env=env, obs=obs)
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = session
    _get_next_question(session)
    return session_id


def _get_session(session_id: str) -> QuizSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("Session not found")
    return session


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_quiz():
    session_id = _start_session()
    return redirect(url_for("quiz", session_id=session_id))


@app.route("/quiz/<session_id>", methods=["GET", "POST"])
def quiz(session_id: str):
    try:
        session = _get_session(session_id)
    except KeyError:
        return redirect(url_for("index"))

    if request.method == "POST":
        user_answer = request.form.get("answer", "")
        current_question = session.current_question or {}
        expected_answer = current_question.get("answer", "")
        is_correct = _evaluate_answer(expected_answer, user_answer)

        session.history.append(
            {
                "prompt": current_question.get("prompt", ""),
                "skill": current_question.get("skill", ""),
                "difficulty": current_question.get("difficulty", ""),
                "question_type": current_question.get("question_type", ""),
                "choices": current_question.get("choices"),
                "expected_answer": expected_answer,
                "user_answer": user_answer,
                "is_correct": is_correct,
                "reward": current_question.get("reward", 0.0),
            }
        )

        if session.done:
            return redirect(url_for("summary", session_id=session_id))

        _get_next_question(session)
        if session.done:
            return redirect(url_for("summary", session_id=session_id))

    if session.done:
        return redirect(url_for("summary", session_id=session_id))

    return render_template(
        "quiz.html",
        session_id=session_id,
        question=session.current_question,
        steps_remaining=MAX_QUESTIONS - session.steps,
        history=session.history,
        agent_available=AGENT is not None,
    )


@app.route("/summary/<session_id>")
def summary(session_id: str):
    try:
        session = _get_session(session_id)
    except KeyError:
        return redirect(url_for("index"))

    total_questions = len(session.history)
    correct_answers = sum(1 for entry in session.history if entry["is_correct"])
    accuracy = (correct_answers / total_questions * 100.0) if total_questions else 0.0

    return render_template(
        "summary.html",
        session_id=session_id,
        history=session.history,
        total_reward=session.total_reward,
        total_questions=total_questions,
        correct_answers=correct_answers,
        accuracy=accuracy,
    )


@app.route("/reset/<session_id>", methods=["POST"])
def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
