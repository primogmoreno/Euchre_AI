Hello all,

My name is Primo Moreno and this is a project inspired by playing the classic Midwestern card game Euchre with my friends at Augustana College.

The creation of this project was assisted with Anthropic's Claude AI chatbot.

This is a project set up to train euchre machine learning models to play euchre.

euchre_ai/
├── game/           # Game simulator (environment)
├── agents/         # Player implementations
├── model/          # Neural network architecture
├── training/       # RL training loop
├── evaluation/     # Testing and tournaments
├── checkpoints/    # Saved models
└── scripts/        # Entry points

Activate virtual environment:
bash: .\.venv\Scripts\Activate.ps1 

Setup
bash: pip install -r requirements.txt

Usage
Train a model:
bash: python scripts/train.py

Play against the AI:
bash: python scripts/play.py

Evaluate agents:
bash: python scripts/evaluate.py

Run UI:
bash: python ui/app.py

Game Rules
Euchre is a 4-player trick-taking card game played with a 24-card deck (9-A in each suit).
Teams of 2 sit across from each other. The team that calls trump must win 3+ tricks
to score points.
Key mechanics:

Jack of trump (right bower) is highest
Jack of same color (left bower) is second highest and counts as trump
Must follow led suit if possible

Console Commands to set up dependincies:
    Setting up virtual environment:
    .\venv\Scripts\Activate
    pip install -r requirements.txt
    Ctrl + Shift + P

    Select Python Interpreter 
