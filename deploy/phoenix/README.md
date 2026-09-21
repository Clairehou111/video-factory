# Local Phoenix

Phoenix is an optional view over the factory's local observability ledger. The
factory continues normally when this service or its Python packages are absent.

Install the exporter extras and start the pinned local service:

```sh
python -m pip install -e '.[observability]'
docker compose -f deploy/phoenix/compose.yaml up -d
```

Copy the variables from `.env.example` into the process environment to enable
export. Open <http://127.0.0.1:6006> to inspect traces. Data persists under
`workspace/phoenix`; the redacted source-of-truth events remain in
`workspace/observability/events.jsonl`.

Stop the UI and collector without deleting persisted data:

```sh
docker compose -f deploy/phoenix/compose.yaml down
```
