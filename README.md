# CS229 Pool Game AI

## Usage

```bash
# (Optional) Create & activate virtual environment
$ virtualenv .venv
$ source .venv/bin/activate

# Install package
$ python -m pip install -r requirements.txt

# Run game
$ python -m src.game.main

# Run training, ALGO = q-table | dqn | ac
$ python -m src.model.train [--balls BALLS] [--algo ALGO] [--visualize] output_model
```

## Problem Formulation

- States
    - `[(x, y)]`: list of (x, y) coordinates of the balls; white ball first
        - Coordinate
            - Continuous range: `[0, 1000]`
            - Discrete range (Q-table only): `[0, 19]`
- Actions
    - Angle
        - Continuous range: `[0, 1]`
        - Discrete range (Q-table only): `[0, 17]` 
    - Force
        - Continuous range: `[0, 1]`
        - Discrete range (Q-table only): `[0, 4]` 
- Rewards
    - If pocket the balls, reward += 5 for each pocketed ball
    - If no ball is hit, reward += -1
