# FearGreedIndex

FearGreedIndex is a lightweight market-sentiment monitor built with Python, GitHub Actions, GitHub Issues, Google Sheets, and a versioned CSV dataset. It retrieves the latest **CNN Fear & Greed Index** reading directly from CNN's JSON feed, classifies the reading against configurable thresholds, and automates three jobs:

1. **Alerting:** Create, reopen, update, or close GitHub Issues when the index enters or leaves extreme fear or extreme greed.
2. **Intraday history:** Record changed index values in Google Sheets for later analysis and visualization.
3. **Daily history:** Maintain one Fear & Greed value per UTC date in `data/fear_greed_daily.csv` and commit changes to the repository.

The project is a monitoring and data-collection tool. It does not generate trading signals or investment recommendations.

## How it works

```text
CNN Fear & Greed JSON feed
          |             |
          v             v
   FearGreed.py   FearGreedHistory.py
     /      \             |
    /        \            v
GitHub      Google     Daily CSV
Issues      Sheets    in repository
alerts      history
```

`FearGreed.py` handles current readings, alerts, and Google Sheets updates. It:

- Fetches the current score, rating, and source timestamp from CNN.
- Classifies the score as `low`, `normal`, or `high`.
- Prints human-readable or JSON output.
- Exposes values to later GitHub Actions steps.
- Optionally inserts a new row into Google Sheets.
- Skips duplicate sheet entries when the rounded whole-number score has not changed.

`FearGreedHistory.py` handles the daily repository dataset. It:

- Requests CNN history beginning on `2021-02-01` by default.
- Converts historical timestamps into UTC dates.
- Keeps only the latest observation for each date.
- Merges fetched rows with the existing CSV so older stored history is not lost if the API later returns a shorter range.
- Rewrites the CSV in chronological order only when its content changes.

The default alert thresholds are:

- `low`: score less than or equal to `25`
- `high`: score greater than or equal to `75`
- `normal`: score between those thresholds

## Repository structure

| Path | Purpose |
| --- | --- |
| `FearGreed.py` | Current-value fetcher, threshold classifier, GitHub Actions output writer, and Google Sheets updater |
| `FearGreedHistory.py` | Daily history fetcher, deduplicator, merger, and CSV writer |
| `data/fear_greed_daily.csv` | Versioned one-row-per-day CNN Fear & Greed dataset |
| `.github/workflows/alert.yml` | Scheduled GitHub Issues alert workflow |
| `.github/workflows/spreasheet.yml` | Scheduled Google Sheets history workflow |
| `.github/workflows/history.yml` | Scheduled daily CSV refresh and commit workflow |
| `fearandgreed.ipynb` | Earlier exploratory notebook and prototype; production automation uses the Python scripts |
| `test_feargreed.py` | Unit tests for CNN record parsing |
| `requirements.txt` | Core runtime dependency list |

> The workflow filename `spreasheet.yml` is intentionally documented exactly as it currently exists in the repository.

## Requirements

- Python 3.10 or newer
- Python 3.12 is used by the GitHub Actions workflows

Create a local environment and install the core dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Google Sheets support requires two additional packages:

```bash
python -m pip install gspread google-auth
```

Do not rely on a committed virtual-environment directory. Create a fresh environment for each machine or CI runner.

## Current-reading usage

### Print the latest reading

```bash
python FearGreed.py
```

Example:

```text
Fear & Greed Index: 39.4286 (Fear, 2026-07-24)
Alert status: normal
```

### Return machine-readable JSON

```bash
python FearGreed.py --json
```

Example structure:

```json
{
  "checked_at_utc": "2026-07-25T00:05:00+00:00",
  "date": "2026-07-24",
  "value": 39.4285714285714,
  "rating": "fear",
  "alert_type": "normal",
  "sheet_updated": false
}
```

### Use different alert thresholds

```bash
python FearGreed.py --low 20 --high 80
```

Thresholds must be between `0` and `100`, and the low threshold must be lower than the high threshold.

### Write GitHub Actions outputs

```bash
python FearGreed.py --github-output
```

When `GITHUB_OUTPUT` is available, the script exports:

- `date`
- `value`
- `rating`
- `alert_type`
- `issue_title`
- `issue_body`

## Google Sheets usage

```bash
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'
python FearGreed.py --update-sheet
```

By default, the script opens:

- Spreadsheet: `FearAndGreed`
- Worksheet: `Sheet1`

Override them with command-line options:

```bash
python FearGreed.py \
  --update-sheet \
  --spreadsheet "My Spreadsheet" \
  --worksheet "History"
```

You can also set:

```bash
export GOOGLE_SPREADSHEET_NAME="My Spreadsheet"
export GOOGLE_WORKSHEET_NAME="History"
```

The sheet uses these columns:

| Date | Time | Value | Site Updated |
| --- | --- | --- | --- |
| UTC check date | UTC check time | Rounded index value | Age of CNN's source reading |

New records are inserted into row 2, directly below the header. By default, no row is added when the newest recorded whole-number value matches the current rounded value.

To record every successful check, including unchanged values:

```bash
python FearGreed.py --update-sheet --append-unchanged
```

## Daily CSV usage

Build or refresh the repository CSV locally:

```bash
python FearGreedHistory.py
```

The default output is:

```text
data/fear_greed_daily.csv
```

Its columns are:

| Column | Description |
| --- | --- |
| `Date` | UTC date in `YYYY-MM-DD` format |
| `Value` | CNN Fear & Greed score from 0 to 100 |
| `Rating` | CNN sentiment label |
| `Source Timestamp UTC` | Timestamp of the observation retained for that date |

If CNN returns multiple observations for a date, the script keeps the one with the latest source timestamp. Existing rows are merged with newly fetched rows, and the file is sorted oldest to newest.

Use a different starting date or output path:

```bash
python FearGreedHistory.py \
  --start-date 2021-02-01 \
  --output data/fear_greed_daily.csv
```

The automated dataset intentionally starts on **2021-02-01**, the reliable boundary for the current CNN history feed. Older archived compilations exist, but they are not mixed into this CNN-only CSV.

## Command-line options

### `FearGreed.py`

| Option | Description |
| --- | --- |
| `--json` | Print a JSON result instead of human-readable output |
| `--github-output` | Write values to the GitHub Actions output file |
| `--update-sheet` | Insert the reading into Google Sheets |
| `--append-unchanged` | Insert a row even when the rounded value is unchanged |
| `--timeout SECONDS` | Set the CNN request timeout; default is `30` |
| `--low VALUE` | Set the low-alert threshold; default is `25` |
| `--high VALUE` | Set the high-alert threshold; default is `75` |
| `--spreadsheet NAME` | Select the Google spreadsheet |
| `--worksheet NAME` | Select the worksheet |

### `FearGreedHistory.py`

| Option | Description |
| --- | --- |
| `--start-date YYYY-MM-DD` | First date requested from CNN; default is `2021-02-01` |
| `--output PATH` | CSV output path; default is `data/fear_greed_daily.csv` |
| `--timeout SECONDS` | Set the CNN request timeout; default is `30` |

Both scripts use these exit codes:

- `0`: success
- `1`: fetch, parsing, file, credential, or Google Sheets failure
- `2`: invalid command-line values

## GitHub Actions automation

### Threshold alert workflow

File: `.github/workflows/alert.yml`

Schedule:

```cron
30 11-22 * * 1-5
```

This runs hourly at minute `30`, from `11:30` through `22:30` UTC, Monday through Friday.

Approximate U.S. Eastern coverage:

- During daylight saving time: `7:30 AM` through `6:30 PM` EDT
- During standard time: `6:30 AM` through `5:30 PM` EST

The workflow uses default thresholds of `25` and `75` and manages two persistent issues:

- `Fear & Greed LOW Alert`
- `Fear & Greed HIGH Alert`

Behavior:

- When the reading is normal, open low/high alert issues are closed.
- When the reading enters an alert range, the corresponding issue is created or reopened.
- The first alert for a new data date creates a notification.
- A same-day notification is added only after a movement of at least `5` points from the last notified value.
- The issue body is updated with the latest notification state.
- Alert issues are assigned to `george962`.

Thresholds, movement size, and assignee are configured in `alert.yml`.

### Google Sheets workflow

File: `.github/workflows/spreasheet.yml`

Schedule:

```cron
3-59/5 * * * 1-5
```

This runs at minutes `03, 08, 13, ... 58` of every hour, Monday through Friday. It runs throughout the full UTC day; it is **not restricted to stock-market hours**.

The workflow installs the Google client libraries and runs:

```bash
python FearGreed.py --update-sheet
```

Because unchanged rounded values are skipped, frequent checks do not necessarily create frequent rows.

### Daily CSV workflow

File: `.github/workflows/history.yml`

Schedule:

```cron
45 23 * * 1-5
```

This runs at `23:45 UTC`, Monday through Friday:

- `7:45 PM` during U.S. Eastern daylight saving time
- `6:45 PM` during U.S. Eastern standard time

It can also be started manually and runs when the history script or its workflow definition is changed on `main`.

The workflow:

1. Fetches CNN history beginning on `2021-02-01`.
2. Updates `data/fear_greed_daily.csv` with one row per UTC date.
3. Makes no commit when the CSV is unchanged.
4. Commits and pushes only the CSV when new or revised daily data exists.

The workflow uses `contents: write` so the repository's `GITHUB_TOKEN` can push the generated CSV.

## Google Sheets configuration

1. Create a Google Cloud service account.
2. Enable the Google Sheets API and Google Drive API for the project.
3. Create and download a JSON key for the service account.
4. Create the spreadsheet and worksheet, or change the configured names.
5. Share the spreadsheet with the service account's `client_email` as an editor.
6. In the GitHub repository, create an Actions secret named:

```text
GOOGLE_SERVICE_ACCOUNT_JSON
```

7. Paste the complete service-account JSON as the secret value.

Never commit the service-account JSON or other credentials to the repository.

## Development notes

Run the unit tests with:

```bash
python -m unittest -v
```

The current `test_feargreed.py` expectations reflect an older three-item return value from `parse_record()`. The production function now also returns the parsed source timestamp, so the tests should be updated before treating the suite as passing.

## Historical coverage

The automated CSV uses only the current CNN endpoint and starts on `2021-02-01`. Publicly available archived compilations can extend the series to `2011-01-03`, but those older values come from separate historical datasets and may differ in methodology or accuracy. They are intentionally excluded from the default CSV to preserve a consistent source.

## Limitations

- The project depends on CNN's current JSON endpoint and response structure, which may change.
- The CNN-only daily CSV does not include pre-February-2021 archived values.
- GitHub Actions cron schedules use UTC and do not automatically exclude U.S. market holidays.
- Scheduled GitHub Actions runs may begin later than the exact cron minute.
- The index's source timestamp can be older than the time at which the workflow checks it.
- The Fear & Greed Index is one sentiment measure and should not be used by itself as an investment decision rule.
