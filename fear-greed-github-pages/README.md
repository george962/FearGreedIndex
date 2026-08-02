# Fear & Greed GitHub Pages Dashboard

This repository builds a static market-research dashboard with GitHub Actions and
deploys it to GitHub Pages.

The page compares Fear & Greed observations with subsequent SPY returns and displays:

- a current rule-based research action;
- historical positive-return probability;
- average and median forward returns;
- worst historical outcome and subsequent drawdown;
- threshold and sudden-drop event studies;
- current historical analogs;
- data-quality warnings; and
- an automatic browser update check.

## Important privacy note

A private repository does **not** make an ordinary GitHub Pages site private.
The generated dashboard should be treated as publicly accessible unless your
organization uses a GitHub Enterprise Cloud private Pages configuration.

Do not put brokerage balances, private positions, credentials, tokens, or other
sensitive information in the dashboard inputs or generated output.

## Repository contents

```text
.
├── .github/workflows/deploy-pages.yml
├── data/fear_greed.tsv
├── scripts/build_dashboard.py
├── scripts/update_fear_greed.py
├── site/.gitkeep
├── config.json
├── requirements.txt
├── .gitignore
└── README.md
```

## First-time GitHub setup

1. Copy the files into the root of your private repository.
2. Confirm the default branch is named `main`. If it is not, edit the `push.branches`
   value in `.github/workflows/deploy-pages.yml`.
3. Commit and push the files.
4. Open the repository on GitHub.
5. Go to **Settings → Pages**.
6. Under **Build and deployment**, set **Source** to **GitHub Actions**.
7. Open **Actions** and run **Build and deploy market dashboard** manually once.
8. After deployment finishes, the workflow's deployment job shows the Pages URL.

Your GitHub plan must support Pages for private repositories. The published site is
normally public even though the repository is private.

## Scheduled behavior

The workflow runs:

- whenever `main` changes;
- whenever you start it manually; and
- at 23:30 UTC every weekday.

That corresponds to:

- 5:30 PM Central Standard Time; or
- 6:30 PM Central Daylight Time.

GitHub scheduled workflows may be delayed. This project is intended for research and
end-of-day decisions, not real-time order execution.

## How the open browser page updates

The generated page requests `version.json` every five minutes. When a newly deployed
build has a different build ID, the page reloads automatically.

Change the interval in `config.json`:

```json
"refresh_seconds": 300
```

This does not make the underlying data real-time. It only causes the browser to notice
a newly deployed GitHub Pages build.

## Fear & Greed data

The included file is:

```text
data/fear_greed.tsv
```

Expected columns:

```text
Date    Time    Value
7/29/2026    21:04:20    32
```

Extra columns are allowed.

### Automatic source option

The project does not hard-code scraping of a third-party Fear & Greed webpage because
page layouts and terms can change.

To refresh the data automatically, provide a stable CSV/TSV/TXT URL that you control:

1. Open **Settings → Secrets and variables → Actions**.
2. Create a repository secret named:

```text
FEAR_GREED_SOURCE_URL
```

3. Set it to a URL returning Date, optional Time, and Value columns.

During each workflow, `scripts/update_fear_greed.py` downloads that source and merges
it with the repository copy for the current build.

Because the workflow deploys a Pages artifact without committing generated data,
downloaded observations do not persist into the repository after the runner ends.
For durable history, either:

- update `data/fear_greed.tsv` in the repository periodically;
- change the workflow to commit updated data; or
- point the secret to a source that itself retains the full historical dataset.

A full-history source is the cleanest option.

## Market data

`config.json` defaults to:

```json
"ticker": "SPY"
```

SPY is used because it is a tradable ETF proxy. Change it to `^GSPC` to use the S&P 500
price index.

Prices are downloaded through `yfinance`. Yahoo Finance access is unofficial and can
occasionally fail or change. When the build fails, GitHub Pages retains the last
successful deployment.

## Action meanings

The dashboard can display:

- **BUY GRADUALLY** — similar historical observations had favorable five-day odds,
  positive average returns, and positive excess return versus baseline.
- **WAIT ON EXTRA BUYING** — similar historical observations were more often followed
  by additional short-term weakness.
- **NEUTRAL** — historical evidence was mixed.
- **INSUFFICIENT EVIDENCE** — too few completed analogs were available.

The dashboard never recommends an all-in trade or automatic sale.

## Configuration

Edit `config.json` to adjust:

- ticker;
- daily Fear & Greed aggregation;
- minimum analog sample;
- cooldown between events;
- forward-return horizons;
- threshold tests;
- analog matching bands; and
- browser update interval.

Valid daily aggregation values are:

- `last`
- `minimum`
- `average`

## Optional local validation

Local execution is not required for normal operation, but it is useful before pushing
major changes:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python scripts/update_fear_greed.py
python scripts/build_dashboard.py
```

Then open `site/index.html`.

## Limitations

- Fear & Greed may respond to the same price decline it appears to predict.
- A small number of independent fear episodes can create misleading win rates.
- Missing dates and long gaps can bias the analog set.
- The index methodology or source may change.
- Yahoo Finance data may be delayed, revised, unavailable, or temporarily blocked.
- GitHub Actions schedules can run late.
- Historical performance does not guarantee future performance.

This is a research dashboard, not personalized financial advice.
