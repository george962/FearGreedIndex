# FGI-001: publication and provenance preflight

This checklist documents **owner decisions** and independent checks needed before making the dashboard easily discoverable or declaring the site public-launch ready. It does not authorize a change to deployment settings or to the existing `noindex,nofollow` metadata.

- [ ] Confirm the canonical public GitHub Pages URL and manually load both `index.html` and `decision_history.html` after a permitted deployment. The repository's `has_pages` setting alone is not proof of a reachable page.
- [ ] Verify the current Fear & Greed data vendor/source, contractual or Terms-of-Use permissions, attribution, and whether raw observations or derivative CSVs may be redistributed publicly. `FEAR_GREED_SOURCE_URL` must not be revealed if it is a secret.
- [ ] Verify rights and required attribution for SPX/SPY historical price data, chart exports, and any third-party branding/icons. Yahoo Finance/yfinance is unofficial, and an API being retrievable does not itself grant publication rights.
- [ ] Inspect `analysis.json`, `historical_decisions.json`, charts, CSVs and `v3_challenger.json` for restricted material, credentials, personal positions and identifiable account data.
- [ ] Confirm that all publication dates and "latest" labels reflect the source observation or report-generation time, rather than implying real-time streaming.
- [ ] Manually test desktop and mobile widths, keyboard-only navigation, focus visibility, chart alternatives, download links and basic screen-reader page structure.
- [ ] Run the documented root tests, v3 tests, original dashboard build, and semantic parity comparison using the same local baseline data. Capture actual exit codes and output diffs.
- [ ] Request owner approval before disabling `noindex`, deploying, altering a credential/Pages setting, or expanding the public dataset.

## Known FGI-001 blocker record (2026-10-09)

At preparation time, the canonical live Pages address, access rights for upstream data/redistribution, and owner approval to enable search indexing were unverified. Therefore FGI-001 must remain `PARTIAL` until those checks can be performed and evidence is recorded.
