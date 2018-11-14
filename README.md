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

# Run training, algo = q-table | dqn | ac
$ python -m src.model.train --balls <num-of-balls> --algo <algorithm>
```
