<<<<<<< HEAD
# FearGreedIndex

A small Python script that prints the latest CNN Fear & Greed Index value
without opening a browser.

MacroMicro chart 50108 tracks CNN's Fear & Greed Index. The script queries
CNN's underlying JSON feed directly, avoiding MacroMicro's interactive
Cloudflare and login checks.

## Setup

```bash
python3 -m venv feargreed_env
source feargreed_env/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
python FearGreed.py
```

Example output:

```text
Fear & Greed Index: 39.4286 (Fear, 2026-07-24)
```

For machine-readable output:

```bash
python FearGreed.py --json
```

## Tests

```bash
python -m unittest -v
```
=======
# FearGreedIndex
>>>>>>> 12ea024e8f8dd281c61a8870ce64961ce11c5784
