# stefie-piper-server

Save-time Piper TTS for the Stefie mobile app. Phone POSTs complete text,
server returns spoken WAV. Port **7777**. Successor to the old `~/piper`
server on port 8080 (kept running untouched until cutover is verified).

## Deploy on the Oracle box

```bash
cd ~/stefie-piper-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# voices: v1 needs at least the German pair
cp ~/piper/de_DE-thorsten-medium.onnx voices/
cp ~/piper/de_DE-thorsten-medium.onnx.json voices/
# later: add en/es/fr/sw pairs from the stefie desktop repo language_model/

uvicorn server:app --host 0.0.0.0 --port 7777 --workers 2
```

Ask for **TCP 7777 ingress** on the Oracle VM security list / NSG.

## Verify

```bash
curl -s localhost:7777/health
curl -s "localhost:7777/generate?text=hallo%20welt&lang=de" -o t.wav
curl -s -X POST localhost:7777/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Willkommen am Flughafen.","lang":"de"}' -o t2.wav
# concurrency: two POSTs at once must both succeed (no event-loop block)
```

## Mobile contract (v1)

```
POST /synthesize { "text": "...", "lang": "de" }
  200 -> audio/wav bytes (+ X-Synth-Ms, X-Audio-Bytes, X-Voice-Model)
  404 -> { error } unknown lang / model file missing
  413 -> { error } text over cap, split per paragraph and retry
```

Limits: `MAX_CHARS = 3000` (POST), `LEGACY_MAX_CHARS = 500` (GET).
Timeout on mobile: 60–120s. If long sessions regularly hit 413,
graduate to job-id polling (`POST -> { job_id }`, `GET /result/{job_id}`)
before reaching for a broker — no Celery needed at this scale.
