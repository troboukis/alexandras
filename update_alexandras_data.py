#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: PyMuPDF (import name: fitz)") from exc

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: openai") from exc


DIAVGEIA_SEARCH_URL = "https://diavgeia.gov.gr/luminapi/api/search"
DEFAULT_CSV_PATH = Path("data_alexandras.csv")
DEFAULT_PDF_DIR = Path("data/pdfs")
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
PAGE_SIZE = 100
DEFAULT_LOG_DIR = Path("logs")

TERMS = [
    "προσφυγικών πολυκατοικιών στη Λεωφόρο Αλεξάνδρας",
    "ΔΙΑΤΗΡΗΤΕΕΣ ΟΚΤΩ ΠΟΛΥΚΑΤΟΙΚΙΕΣ ΤΗΣ ΛΕΩΦΟΡΟΥ ΑΛΕΞΑΝΔΡΑΣ",
    "ΠΡΟΣΦΥΓΙΚΩΝ ΠΟΛΥΚΑΤΟΙΚΙΩΝ ΣΤΗ ΛΕΩΦΟΡΟ ΑΛΕΞΑΝΔΡΑΣ",
    "Προσφυγικών Πολυκατοικιών στη Λεωφ. Αλεξάνδρας",
    "ΠΡΟΣΦΥΓΙΚΩΝ ΠΟΛΥΚΑΤΟΙΚΙΩΝ ΣΤΗ Λ. ΑΛΕΞΑΝΔΡΑΣ",
    "ΠΡΟΣΦΥΓΙΚΕΣ ΠΟΛΥΚΑΤΟΙΚΙΕΣ ΤΗΣ ΛΕΩΦ.ΑΛΕΞΑΝΔΡΑΣ",
]


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_log_file(log_file: Path | None) -> tuple[Any, Any, Any]:
    if log_file is None:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = DEFAULT_LOG_DIR / f"update_alexandras_{timestamp}.log"
    else:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = log_file.open("a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, fh)
    sys.stderr = TeeStream(original_stderr, fh)
    print(f"Logging to: {log_file}")
    return fh, original_stdout, original_stderr


def teardown_log_file(log_ctx: tuple[Any, Any, Any] | None) -> None:
    if not log_ctx:
        return
    fh, original_stdout, original_stderr = log_ctx
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    fh.close()


def load_dotenv_if_present(dotenv_path: Path = Path(".env")) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def build_query_string() -> str:
    quoted = ", ".join(json.dumps(t, ensure_ascii=False) for t in TERMS)
    return f"q:[{quoted}]"


def build_params(page: int) -> dict[str, Any]:
    # Focus on Περιφέρεια Αττικής (organizationUid 5002), same as the notebook's main search path.
    return {
        "fq": 'organizationUid:"5002"',
        "page": page,
        "q": build_query_string(),
        "sort": "relative",
        "size": PAGE_SIZE,
    }


def fetch_all_decisions(session: requests.Session, sleep_s: float = 0.2) -> list[dict[str, Any]]:
    print("Fetching Diavgeia search page 1...")
    first = session.get(
        DIAVGEIA_SEARCH_URL,
        params=build_params(0),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    first.raise_for_status()
    payload = first.json()
    total = int(payload.get("info", {}).get("total", 0))
    decisions = list(payload.get("decisions", []))
    total_pages = max(1, math.ceil(total / PAGE_SIZE)) if total else 1

    print(f"Diavgeia search returned total={total}, page_size={PAGE_SIZE}, pages={total_pages}")

    for page in range(1, total_pages):
        print(f"Fetching Diavgeia page {page + 1}/{total_pages}...")
        resp = session.get(
            DIAVGEIA_SEARCH_URL,
            params=build_params(page),
            headers={"Accept": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        page_payload = resp.json()
        decisions.extend(page_payload.get("decisions", []))
        time.sleep(sleep_s)

    return decisions


def extract_row(decision: dict[str, Any]) -> dict[str, Any]:
    organization = decision.get("organization") or {}
    decision_type = decision.get("decisionType") or {}
    ada = (decision.get("ada") or "").strip()
    return {
        "label": organization.get("label", ""),
        "subject": (decision.get("subject") or "").strip(),
        "ada": ada,
        "issueDate": decision.get("issueDate", ""),
        "documentUrl": decision.get("documentUrl", ""),
        "documentType": decision.get("documentType", ""),
        "decisionType": decision_type.get("label", ""),
        "file": ada,
    }


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    expected_cols = [
        "label",
        "subject",
        "ada",
        "issueDate",
        "documentUrl",
        "documentType",
        "decisionType",
        "file",
        "summary",
    ]
    if not csv_path.exists():
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df[expected_cols]


def metadata_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("label", "")),
        str(row.get("subject", "")),
        str(row.get("ada", "")),
        str(row.get("issueDate", "")),
        str(row.get("documentUrl", "")),
        str(row.get("documentType", "")),
        str(row.get("decisionType", "")),
        str(row.get("file", "")),
    )


def download_pdf(session: requests.Session, document_url: str, pdf_dir: Path, ada: str) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_name = Path(urlparse(document_url).path).name
    filename = parsed_name if parsed_name and parsed_name.lower().endswith(".pdf") else f"{ada}.pdf"
    pdf_path = pdf_dir / filename
    if pdf_path.exists():
        print(f"  PDF cache hit: {pdf_path.name}")
        return pdf_path

    print(f"  Downloading PDF for {ada}...")
    resp = session.get(document_url, timeout=60, allow_redirects=True)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "pdf" not in content_type and not resp.content.startswith(b"%PDF"):
        raise ValueError(f"URL did not return a PDF for {ada}: {document_url}")

    pdf_path.write_bytes(resp.content)
    print(f"  Saved PDF: {pdf_path}")
    return pdf_path


def pdf_text(path: Path, max_chars: int = 120_000, max_chars_per_page: int = 10_000) -> str:
    out: list[str] = []
    used = 0
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            if not text:
                continue
            if len(text) > max_chars_per_page:
                text = text[:max_chars_per_page] + " ..."
            block = f"[PAGE {i}] {text}"
            remaining = max_chars - used
            if remaining <= 0:
                break
            if len(block) > remaining:
                out.append(block[:remaining] + " ...")
                break
            out.append(block)
            used += len(block) + 1
    return "\n".join(out)


def summarization_prompt(text: str) -> str:
    terms_json = json.dumps(TERMS, ensure_ascii=False, indent=2)
    return f"""
Όροι αναφοράς:
{terms_json}

Κείμενο εγγράφου (ενδέχεται περικομμένο):
{text}

Επέστρεψε μόνο έγκυρο JSON (χωρίς markdown, χωρίς σχόλια) με ακριβώς αυτά τα keys:
relevance, description, table, companies, pages, extras

Κανόνες:
1. relevance: "YES" μόνο αν το έγγραφο αφορά ουσιαστικά τα προσφυγικά της Λεωφόρου Αλεξάνδρας. Αλλιώς "NO".
2. description: σύντομη περίληψη έως 100 λέξεις στα ελληνικά.
3. table: λίστα από γραμμές οικονομικών στοιχείων μόνο αν αφορούν τα προσφυγικά της Λ. Αλεξάνδρας, αλλιώς [].
4. companies: λίστα εταιρειών/αναδόχων/νομικών προσώπων που σχετίζονται άμεσα με το θέμα.
5. pages: λίστα αριθμών σελίδων όπου υπάρχουν σχετικές αναφορές.
6. extras: αν υπάρχουν πρακτικά/συνεδριάσεις, σύντομη περιγραφή. Αλλιώς "".
7. Μην προσθέσεις κανένα άλλο key.
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty OpenAI response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in OpenAI response")

    candidate = text[start : end + 1]
    # Remove common markdown fences if the model ignored instructions.
    candidate = candidate.replace("```json", "").replace("```", "").strip()
    return json.loads(candidate)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(value).strip()] if str(value).strip() else []


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    seq = value if isinstance(value, list) else [value]
    pages: list[int] = []
    for item in seq:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            n = int(item)
            if n > 0:
                pages.append(n)
            continue
        if isinstance(item, str):
            for token in re.findall(r"\d+", item):
                n = int(token)
                if n > 0:
                    pages.append(n)
    return sorted(set(pages))


def _trim_to_words(text: str, max_words: int = 100) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,.;:") + "..."


def clean_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    relevance_raw = str(summary.get("relevance", "")).strip().upper()
    cleaned["relevance"] = "YES" if relevance_raw == "YES" else "NO"

    description = str(summary.get("description", "") or "").strip()
    cleaned["description"] = _trim_to_words(description, 100)

    table = _as_string_list(summary.get("table"))
    companies = _as_string_list(summary.get("companies"))
    pages = _as_int_list(summary.get("pages"))
    extras = str(summary.get("extras", "") or "").strip()

    # Remove obvious duplicates while preserving order.
    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = re.sub(r"\s+", " ", item).strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(re.sub(r"\s+", " ", item).strip())
        return out

    cleaned["table"] = dedupe(table)
    cleaned["companies"] = dedupe(companies)
    cleaned["pages"] = pages
    cleaned["extras"] = extras

    if cleaned["relevance"] == "NO":
        # Keep the schema but avoid carrying irrelevant noisy details.
        cleaned["table"] = []
        cleaned["companies"] = []
        cleaned["pages"] = []

    return cleaned


def summarize_pdf(client: OpenAI, model: str, pdf_path: Path) -> tuple[str, dict[str, Any]]:
    print(f"  Extracting text from {pdf_path.name}...")
    text = pdf_text(pdf_path)
    if not text:
        raw = json.dumps(
            {
                "relevance": "NO",
                "description": "Δεν κατέστη δυνατή η εξαγωγή κειμένου από το PDF.",
                "table": [],
                "companies": [],
                "pages": [],
                "extras": "",
            },
            ensure_ascii=False,
        )
        return raw, json.loads(raw)

    print(f"  Sending summary request to OpenAI ({model})...")
    resp = client.responses.create(
        model=model,
        input=summarization_prompt(text),
    )
    raw = (getattr(resp, "output_text", None) or "").strip()
    print("  OpenAI response received. Cleaning JSON...")
    parsed = extract_json_object(raw)
    cleaned = clean_summary(parsed)
    return raw, cleaned


def clean_existing_summaries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    print(f"Cleaning existing summaries for {len(df)} rows...")
    cleaned_rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        if i == 1 or i % 25 == 0 or i == len(df):
            print(f"  Cleaned existing summaries: {i}/{len(df)}")
        summary_text = str(row.get("summary", "") or "").strip()
        if not summary_text:
            cleaned_rows.append(row.to_dict())
            continue
        try:
            parsed = extract_json_object(summary_text)
            cleaned = clean_summary(parsed)
            row_dict = row.to_dict()
            row_dict["summary"] = json.dumps(cleaned, ensure_ascii=False, indent=2)
            cleaned_rows.append(row_dict)
        except Exception:
            cleaned_rows.append(row.to_dict())
    return pd.DataFrame(cleaned_rows, columns=df.columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update data_alexandras.csv from Diavgeia and summarize new PDFs with OpenAI."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to output CSV.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Directory for downloaded PDFs.")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model for summarization.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write console output to this log file too (default: auto timestamped file in ./logs).",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable file logging for this run.",
    )
    parser.add_argument(
        "--include-irrelevant",
        action="store_true",
        help="Append rows even when relevance is NO (default: only add relevance=YES).",
    )
    parser.add_argument(
        "--clean-existing",
        action="store_true",
        help="Normalize existing JSON summaries in the CSV before saving.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between Diavgeia page requests (seconds).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N new decisions (0 = no limit).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-document processing details.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_ctx = None
    if not args.no_log_file:
        log_ctx = setup_log_file(args.log_file)
    load_dotenv_if_present()
    print("Starting Alexandras dataset update...")
    print(f"CSV path: {args.csv}")
    print(f"PDF dir: {args.pdf_dir}")
    print(f"OpenAI model: {args.model}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set (checked env and .env).", file=sys.stderr)
        teardown_log_file(log_ctx)
        return 2

    existing_df = load_existing_csv(args.csv)
    if args.clean_existing:
        print("Option --clean-existing is ON: existing summary JSON rows will be normalized before save.")
        existing_df = clean_existing_summaries(existing_df)
    else:
        print("Existing rows will NOT be re-analyzed or cleaned (new rows only).")

    print(f"Loaded {len(existing_df)} existing rows from {args.csv}")
    existing_metadata_signatures = {
        metadata_signature(row)
        for row in existing_df.drop(columns=["summary"], errors="ignore").to_dict(orient="records")
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "alexandras-updater/1.0"})
    client = OpenAI(api_key=api_key)
    print("OpenAI client initialized.")

    try:
        decisions = fetch_all_decisions(session, sleep_s=args.sleep)
    except Exception as exc:
        print(f"Failed to fetch Diavgeia data: {exc}", file=sys.stderr)
        teardown_log_file(log_ctx)
        return 1

    metadata_rows: list[dict[str, Any]] = []
    seen_batch_signatures: set[tuple[str, ...]] = set()
    for idx, decision in enumerate(decisions, start=1):
        if idx == 1 or idx % 100 == 0 or idx == len(decisions):
            print(f"Scanning fetched decisions: {idx}/{len(decisions)}")
        row = extract_row(decision)
        sig = metadata_signature(row)
        if not row["ada"]:
            continue
        if sig in seen_batch_signatures or sig in existing_metadata_signatures:
            continue
        seen_batch_signatures.add(sig)
        metadata_rows.append(row)

    if args.limit and args.limit > 0:
        metadata_rows = metadata_rows[: args.limit]

    print(f"New candidate decisions to inspect: {len(metadata_rows)}")
    if metadata_rows:
        print("Beginning PDF + OpenAI processing for new candidates...")
    if not metadata_rows:
        existing_df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print("No new decisions found. CSV left unchanged (except optional summary cleaning).")
        teardown_log_file(log_ctx)
        return 0

    new_rows: list[dict[str, Any]] = []
    for idx, meta in enumerate(metadata_rows, start=1):
        ada = meta["ada"]
        if args.verbose:
            print(f"[{idx}/{len(metadata_rows)}] {ada} - {meta['subject'][:120]}")
        else:
            print(f"Processing {idx}/{len(metadata_rows)}: {ada}")
        try:
            pdf_path = download_pdf(session, meta["documentUrl"], args.pdf_dir, ada)
            _, cleaned_summary = summarize_pdf(client, args.model, pdf_path)
            print(f"  Relevance={cleaned_summary['relevance']} | pages={cleaned_summary.get('pages', [])}")
            if cleaned_summary["relevance"] != "YES" and not args.include_irrelevant:
                print(f"  Skipped {ada} (relevance={cleaned_summary['relevance']})")
                continue

            row = dict(meta)
            row["summary"] = json.dumps(cleaned_summary, ensure_ascii=False, indent=2)
            new_rows.append(row)
            print(f"  Added {ada}. New rows collected: {len(new_rows)}")
        except Exception as exc:
            print(f"Error processing {ada}: {exc}", file=sys.stderr)

    if not new_rows:
        final_df = existing_df.copy()
        final_df.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print("No relevant new rows to append.")
        teardown_log_file(log_ctx)
        return 0

    new_df = pd.DataFrame(new_rows, columns=existing_df.columns)
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    final_df = final_df.drop_duplicates(keep="first")
    print(f"Writing CSV with {len(final_df)} rows...")
    final_df.to_csv(args.csv, index=False, encoding="utf-8-sig")

    print(f"Appended {len(new_df)} new rows. Total rows: {len(final_df)}")
    print(f"Saved updated CSV to {args.csv}")
    teardown_log_file(log_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
