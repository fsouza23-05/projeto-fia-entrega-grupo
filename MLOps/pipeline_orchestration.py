import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def run_step(name: str, command: list[str]) -> dict:
    print(f"\n[PIPELINE] Iniciando etapa: {name}")
    started_at = datetime.now(timezone.utc)

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    ended_at = datetime.now(timezone.utc)
    status = "success" if result.returncode == 0 else "failed"

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return {
        "step": name,
        "command": " ".join(command),
        "status": status,
        "return_code": result.returncode,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
    }


def run_pipeline() -> dict:
    steps = [
        ("data_sanitization", [sys.executable, "DataPipeline/data_sanitization.py"]),
        ("abt_transform", [sys.executable, "DataPipeline/abt_transform.py"]),
        ("train_model", [sys.executable, "Model/train.py"]),
    ]

    execution = []
    for name, cmd in steps:
        step_result = run_step(name, cmd)
        execution.append(step_result)

        if step_result["status"] == "failed":
            summary = {
                "pipeline_status": "failed",
                "executed_steps": execution,
                "failed_step": name,
            }
            print("\n[PIPELINE] Falha detectada. Encerrando fluxo.")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return summary

    summary = {
        "pipeline_status": "success",
        "executed_steps": execution,
    }
    print("\n[PIPELINE] Pipeline concluido com sucesso.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run_pipeline()
