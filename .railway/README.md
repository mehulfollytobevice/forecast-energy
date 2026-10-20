# Railway Infrastructure as Code

This directory replaces the deprecated `railway.toml` / Config as Code files.

| File | Role |
| --- | --- |
| `railway.ts` | Project graph: `web`, `forecast`, `retrain`, and the `snapshots` bucket |

Railway does **not** read this file during a deploy. The CLI does:

```bash
npm install
railway login
railway link
railway config plan
railway config apply
```

Set `PJM_API_KEY` and `OPEN_METEO_API_KEY` on each service (or as shared variables) in the Railway dashboard. The authoring file uses `preserve()` so those values stay on Railway and are never committed.

Do not add `railway.toml` or `railway.json` back. New services cannot opt into Config as Code, and existing files stop being read on 2026-12-01.
