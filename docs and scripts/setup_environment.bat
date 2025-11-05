@echo off
echo ========================================
echo Setting Up Personalized Learning RL
echo ========================================
echo.

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Step 2: Uninstalling conflicting packages...
pip uninstall -y transformers huggingface_hub accelerate protobuf tensorflow
echo.

echo Step 3: Installing core dependencies with exact versions...
pip install protobuf==3.20.3
pip install numpy==1.23.5
pip install tensorflow==2.10.0
echo.

echo Step 4: Installing AI/ML packages...
pip install huggingface_hub==0.16.4
pip install accelerate==0.20.3
pip install transformers==4.30.2
pip install torch==2.0.1
pip install sentence-transformers==2.2.2
echo.

echo Step 5: Installing RL and other dependencies...
pip install stable-baselines3[extra]==2.1.0
pip install gymnasium==0.29.1
pip install pandas matplotlib scikit-learn
pip install bert-score==0.3.13
pip install textstat
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run:
echo   python interactive_quiz.py
echo   python demo_session.py
echo.
pause
