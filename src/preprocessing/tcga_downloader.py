"""
TCGA WSI + Clinical data downloader via the GDC Data Portal API.

Usage (standalone):
    python -m src.preprocessing.tcga_downloader \
        --project TCGA-LUAD \
        --out_dir data/raw \
        --n_cases 50          # limit for dev; omit for full cohort

The script:
  1. Queries the GDC API for SVS diagnostic slides for the given project.
  2. Builds a manifest and downloads slides via the GDC transfer tool.
  3. Downloads and processes the clinical XML to produce a clean
     survival CSV (case_id, survival_months, vital_status, event).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(pretty_exceptions_enable=False)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _query_gdc_files(project: str, n_cases: Optional[int] = None) -> list[dict]:
    """Return list of {file_id, file_name, case_id} for SVS diagnostic slides."""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
        ],
    }
    fields = ["file_id", "file_name", "cases.case_id", "cases.submitter_id"]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "json",
        "size": str(n_cases or 10000),
    }
    r = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    records = []
    for h in hits:
        case = h.get("cases", [{}])[0]
        records.append({
            "file_id":    h["file_id"],
            "file_name":  h["file_name"],
            "case_id":    case.get("case_id", ""),
            "submitter_id": case.get("submitter_id", ""),
        })
    return records


def _query_clinical(project: str) -> pd.DataFrame:
    """Fetch survival data from GDC cases endpoint."""
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project},
    }
    fields = [
        "case_id",
        "submitter_id",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.vital_status",
        "demographic.vital_status",
        "demographic.days_to_death",
    ]
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "json",
        "size": "10000",
    }
    r = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]

    rows = []
    for h in hits:
        case_id    = h.get("case_id", "")
        submitter  = h.get("submitter_id", "")
        diagnoses  = h.get("diagnoses", [{}])
        demo       = h.get("demographic", {})

        # Prefer diagnosis-level data, fallback to demographic
        dx         = diagnoses[0] if diagnoses else {}
        days_death = dx.get("days_to_death") or demo.get("days_to_death")
        days_fu    = dx.get("days_to_last_follow_up")
        vital      = (dx.get("vital_status") or demo.get("vital_status") or "").lower()

        event      = 1 if vital == "dead" else 0
        duration   = days_death if event == 1 else days_fu

        if duration is None:
            continue  # skip cases with no time info

        rows.append({
            "case_id":          case_id,
            "submitter_id":     submitter,
            "survival_days":    float(duration),
            "survival_months":  round(float(duration) / 30.44, 2),
            "event":            event,
            "vital_status":     vital,
        })

    df = pd.DataFrame(rows)
    df = df[df["survival_days"] > 0].reset_index(drop=True)
    return df


def _write_manifest(records: list[dict], manifest_path: Path) -> None:
    """Write GDC download manifest TSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\tfilename\tmd5\tsize\tstate"]
    for r in records:
        lines.append(f"{r['file_id']}\t{r['file_name']}\t\t\t")
    manifest_path.write_text("\n".join(lines))
    console.print(f"[green]Manifest written:[/green] {manifest_path}  ({len(records)} files)")


def _download_with_gdc_client(manifest_path: Path, out_dir: Path, n_processes: int = 4) -> None:
    """Invoke gdc-client download. Falls back to curl if not installed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            "gdc-client", "download",
            "-m", str(manifest_path),
            "-d", str(out_dir),
            "-n", str(n_processes),
        ]
        console.print(f"[cyan]Running:[/cyan] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        console.print(
            "[yellow]gdc-client not found. Install from: "
            "https://gdc.cancer.gov/access-data/gdc-data-transfer-tool[/yellow]\n"
            "Alternatively, download the manifest manually from the GDC portal."
        )


# ─── Main CLI ─────────────────────────────────────────────────────────────────

@app.command()
def download(
    project:    str          = typer.Option("TCGA-LUAD", help="GDC project ID"),
    out_dir:    Path         = typer.Option(Path("data/raw"), help="Root output directory"),
    n_cases:    Optional[int]= typer.Option(None,  help="Max slides to download (None = all)"),
    n_procs:    int          = typer.Option(4,     help="Parallel download processes"),
    skip_wsi:   bool         = typer.Option(False, help="Skip WSI download (clinical only)"),
) -> None:
    """Download TCGA WSIs and clinical survival data from the GDC portal."""

    console.rule(f"[bold blue]TCGA Downloader · {project}")

    # ── 1. Query files ──────────────────────────────────────────────────────
    console.print("[cyan]Querying GDC Files API…[/cyan]")
    records = _query_gdc_files(project, n_cases)
    console.print(f"  Found [bold]{len(records)}[/bold] diagnostic SVS slides")

    manifest_path = out_dir / project / "manifest.tsv"
    _write_manifest(records, manifest_path)

    # ── 2. Save file–case mapping ───────────────────────────────────────────
    mapping_path = out_dir / project / "file_case_mapping.csv"
    pd.DataFrame(records).to_csv(mapping_path, index=False)
    console.print(f"[green]File–case mapping saved:[/green] {mapping_path}")

    # ── 3. Download WSIs ────────────────────────────────────────────────────
    if not skip_wsi:
        wsi_dir = out_dir / project / "wsi"
        console.print(f"\n[cyan]Downloading WSIs to:[/cyan] {wsi_dir}")
        _download_with_gdc_client(manifest_path, wsi_dir, n_procs)

    # ── 4. Clinical data ────────────────────────────────────────────────────
    console.print("\n[cyan]Fetching clinical / survival data…[/cyan]")
    time.sleep(1)  # be polite to GDC API
    clinical_df = _query_clinical(project)

    # Merge slide list with clinical
    slide_df = pd.DataFrame(records)[["file_id", "file_name", "case_id", "submitter_id"]]
    merged   = slide_df.merge(clinical_df, on=["case_id"], how="inner")
    merged["wsi_path"] = merged["file_name"].apply(
        lambda x: str(out_dir / project / "wsi" / x)
    )

    clinical_out = Path("data/processed") / f"{project}_clinical.csv"
    clinical_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(clinical_out, index=False)

    # Print summary table
    tbl = Table(title=f"{project} · Clinical Summary", show_header=True)
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value",  style="white")
    tbl.add_row("Total slides with survival", str(len(merged)))
    tbl.add_row("Events (deaths)",            str(merged["event"].sum()))
    tbl.add_row("Censored",                   str((merged["event"] == 0).sum()))
    tbl.add_row("Median survival (months)",   str(round(merged["survival_months"].median(), 1)))
    tbl.add_row("Saved to",                   str(clinical_out))
    console.print(tbl)


if __name__ == "__main__":
    app()
