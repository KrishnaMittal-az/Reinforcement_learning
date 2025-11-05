@echo off
echo =================================================
echo  Setting Up Personalized Learning RL (v2)
echo =================================================
echo.

echo [1/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [2/4] Uninstalling conflicting packages...
pip uninstall -y numpy pandas tensorflow tensorboard transformers huggingface_hub accelerate protobuf
echo.

echo [3/4] Installing core packages with compatible versions...
echo      - Installing numpy==1.26.4
pip install numpy==1.26.4
echo      - Installing protobuf==3.20.3
pip install protobuf==3.20.3
echo      - Installing tensorflow==2.10.1
pip install tensorflow==2.10.1
echo.

echo [4/4] Installing remaining dependencies...
pip install pandas
pip install huggingface_hub==0.16.4
pip install accelerate==0.20.3
pip install transformers==4.30.2
pip install torch==2.0.1
pip install sentence-transformers==2.2.2
pip install stable-baselines3[extra]==2.1.0
pip install gymnasium==0.29.1
pip install matplotlib scikit-learn
pip install bert-score==0.3.13
pip install textstat
echo.

echo =================================================
echo                  Setup Complete!
echo =================================================
echo.
echo You can now run the quiz:
echo   python interactive_quiz.py
echo.
pause
