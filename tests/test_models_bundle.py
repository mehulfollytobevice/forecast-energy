"""MLflow bundle extract remaps host-absolute artifact URIs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pjm_forecasting.models_bundle import remap_mlflow_artifact_uris


def test_remap_mlflow_artifact_uris_rewrites_foreign_host_paths(tmp_path: Path):
    db_path = tmp_path / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "mlflow"
    artifact_root.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE experiments (experiment_id INTEGER, name TEXT, artifact_location TEXT)"
    )
    conn.execute("CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT)")
    conn.execute(
        "INSERT INTO experiments VALUES (1, 'pjm-AECO', "
        "'file:///Users/other/artifacts/mlflow')"
    )
    conn.execute(
        "INSERT INTO runs VALUES ('6b5209589459453b86955509f5f9d187', "
        "'file:///Users/other/artifacts/mlflow/6b5209589459453b86955509f5f9d187/artifacts')"
    )
    conn.commit()
    conn.close()

    result = remap_mlflow_artifact_uris(db_path, artifact_root)

    expected_root = artifact_root.resolve().as_uri().rstrip("/")
    expected_run = f"{expected_root}/6b5209589459453b86955509f5f9d187/artifacts"
    conn = sqlite3.connect(db_path)
    location = conn.execute("SELECT artifact_location FROM experiments WHERE experiment_id = 1").fetchone()[0]
    uri = conn.execute("SELECT artifact_uri FROM runs").fetchone()[0]
    conn.close()
    assert result["updated_experiments"] == 1
    assert result["updated_runs"] == 1
    assert location == expected_root
    assert uri == expected_run
    assert not uri.startswith("file:///Users/other/")
