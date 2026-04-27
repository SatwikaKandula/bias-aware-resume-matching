#!/usr/bin/env python3
"""Local demo server for optional LLM resume scoring.

The API key is read from OPENAI_API_KEY and is never written into the HTML.
Run from the repository root:

    OPENAI_API_KEY=... python ML_FINAL/demo/server.py

Then open http://127.0.0.1:8000/.
"""

from __future__ import annotations

import json
import os
import io
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
OPENAI_URL = "https://api.openai.com/v1/responses"


SYSTEM_PROMPT = """You are scoring job-resume fit for a classroom fairness demo.
Score only skills, experience, education requirements, projects, and certifications.
Ignore demographic signals such as name, pronouns, gender, race, ethnicity, and
university prestige unless the job materially requires a specific credential."""


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}


def output_text(payload: dict) -> str:
    for item in payload.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content.get("text", "")
    return payload.get("output_text", "")


def call_openai(api_key: str, resume: str, job: str) -> dict:
    body = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Job description:\n{job}\n\nResume:\n{resume}\n\nReturn JSON.",
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "resume_match_score",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }
    request = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return json.loads(output_text(payload))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/extract-pdf":
            self._extract_pdf()
            return

        if self.path != "/api/llm-score":
            self.send_error(404)
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._json({"error": "OPENAI_API_KEY is not set on the local server."}, status=500)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            resume = str(payload.get("resume", "")).strip()
            job = str(payload.get("job", "")).strip()
            if not resume or not job:
                self._json({"error": "Resume and job text are required."}, status=400)
                return
            result = call_openai(api_key, resume, job)
            self._json({"score": float(result["score"]), "rationale": result["rationale"]})
        except urllib.error.HTTPError as exc:
            self._json({"error": exc.read().decode("utf-8", errors="replace")}, status=502)
        except Exception as exc:  # Keep the browser message readable for demos.
            self._json({"error": str(exc)}, status=500)

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _extract_pdf(self) -> None:
        try:
            from pypdf import PdfReader
        except Exception:
            self._json(
                {
                    "error": "Server-side PDF extraction needs pypdf. Run the server with the bundled project Python or install pypdf."
                },
                status=500,
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            pdf_bytes = self.rfile.read(length)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
            if not text:
                self._json({"error": "No selectable text was found in this PDF."}, status=422)
                return
            self._json({"text": text, "pages": len(reader.pages)})
        except Exception as exc:
            self._json({"error": f"Could not extract PDF text: {exc}"}, status=500)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving demo at http://127.0.0.1:{port}/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
