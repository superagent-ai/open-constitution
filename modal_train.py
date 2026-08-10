from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections import deque
from pathlib import Path

import modal

APP_DIR = "/app"
HF_CACHE_DIR = "/cache/huggingface"
OUTPUT_DIR = "/outputs"
APP_NAME = "open-constitution"
REPO_ROOT = Path(__file__).parent

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_CLASSIFIER_MODEL_ID = "answerdotai/ModernBERT-base"
DEFAULT_DATA_PATH = "data/examples.jsonl"
DEFAULT_CLASSIFIER_DATA_PATH = "data/training_data.jsonl"
DEFAULT_OUT_DIR = "probe_out_gemma4"
DEFAULT_CLASSIFIER_OUT_DIR = "classifier_out_modernbert"
DEFAULT_PROMPT = "Explain SQL injection at a high level"

FALLBACK_PROJECT_DEPENDENCIES = [
    "torch>=2.2.0",
    "transformers>=4.55.0",
    "accelerate>=0.33.0",
    "datasets>=2.20.0",
    "safetensors>=0.4.3",
    "scikit-learn>=1.4.0",
    "tqdm>=4.66.0",
    "numpy>=1.26.0",
    "pillow>=12.2.0",
    "torchvision>=0.27.0",
]


def _project_dependencies() -> list[str]:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        return FALLBACK_PROJECT_DEPENDENCIES

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def _read_last_lines(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        return "".join(deque(f, maxlen=max_lines))


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _is_valid_trainer_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "trainer_state.json").exists():
        return False
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def _latest_trainer_checkpoint(out_dir: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if not out_dir.exists():
        return None

    for path in out_dir.iterdir():
        step = _checkpoint_step(path)
        if step is not None and _is_valid_trainer_checkpoint(path):
            checkpoints.append((step, path))

    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def _resolve_resume_checkpoint(
    resume_from_checkpoint: str,
    *,
    remote_out_path: Path,
) -> Path:
    if resume_from_checkpoint == "latest":
        requested_path = None
    else:
        requested_path = Path(
            resume_from_checkpoint
            if resume_from_checkpoint.startswith("/")
            else f"{OUTPUT_DIR}/{resume_from_checkpoint}"
        )
        if _is_valid_trainer_checkpoint(requested_path):
            return requested_path

    latest_path = _latest_trainer_checkpoint(remote_out_path)
    if latest_path is None:
        requested = "latest" if requested_path is None else str(requested_path)
        raise FileNotFoundError(f"No valid checkpoint found for resume request: {requested}")

    if requested_path is None:
        print(f"Resolved latest checkpoint to: {latest_path}")
    else:
        print(f"Requested checkpoint {requested_path} is missing; using latest: {latest_path}")
    return latest_path


def _volume_path(path: str) -> str:
    return path if path.startswith("/") else f"{OUTPUT_DIR}/{path}"


def _print_spawned(name: str, function_call, *, output_path: str) -> None:
    print(f"Spawned {name} function call: {function_call.object_id}")
    print(f"Expected Modal Volume output path: {output_path}")


def _api_artifact_dir(artifact_id: str, artifact_type: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", artifact_id) is None:
        raise ValueError("artifact_id must be a 32-character lowercase hexadecimal UUID")
    if artifact_type not in {"probe", "classifier"}:
        raise ValueError("artifact_type must be 'probe' or 'classifier'")
    return Path(OUTPUT_DIR) / "jobs" / artifact_id / artifact_type


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("open-constitution-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("open-constitution-outputs", create_if_missing=True)
local_env_secret = modal.Secret.from_dict(_dotenv_values(REPO_ROOT / ".env.local"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_project_dependencies())
    .add_local_dir(
        str(REPO_ROOT / "activation_probe_mvp"),
        remote_path=f"{APP_DIR}/activation_probe_mvp",
        copy=True,
    )
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=f"{APP_DIR}/scripts", copy=True)
    .add_local_dir(str(REPO_ROOT / "data"), remote_path=f"{APP_DIR}/data", copy=True)
)

publish_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=1.19.0")
    .env({"PYTHONPATH": APP_DIR})
    .add_local_dir(
        str(REPO_ROOT / "activation_probe_mvp"),
        remote_path=f"{APP_DIR}/activation_probe_mvp",
        copy=True,
    )
)


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={
        OUTPUT_DIR: outputs,
    },
)
def run_build_three_class_classifier_data(
    primary_input_path: str = "data/training_data_classifier.jsonl",
    extra_input_paths: list[str] | None = None,
    out_path: str = f"{OUTPUT_DIR}/training_data_classifier_threeclass.jsonl",
    keep_extra_duplicates: bool = False,
    extra_repeat: int = 1,
) -> str:
    outputs.reload()
    remote_primary_input_path = (
        primary_input_path
        if primary_input_path.startswith("/")
        else f"{APP_DIR}/{primary_input_path}"
    )
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"

    command = [
        "python",
        "-m",
        "scripts.build_three_class_classifier_data",
        "--primary_input_path",
        remote_primary_input_path,
        "--out_path",
        remote_out_path,
        "--extra_repeat",
        str(extra_repeat),
    ]

    for path in extra_input_paths or []:
        remote_extra_path = path if path.startswith("/") else f"{OUTPUT_DIR}/{path}"
        command.extend(["--extra_input_path", remote_extra_path])

    if keep_extra_duplicates:
        command.append("--keep_extra_duplicates")

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        outputs.commit()

    return remote_out_path


@app.function(
    image=image,
    timeout=4 * 60 * 60,
    secrets=[local_env_secret],
    volumes={
        OUTPUT_DIR: outputs,
    },
)
def run_build_benign_recovery_data(
    eval_paths: list[str],
    out_path: str = f"{OUTPUT_DIR}/hard_training/v3_benign_recovery.jsonl",
    limit_per_split: int = 100,
    seed: int = 0,
    generator_model: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
    max_examples: int | None = None,
) -> str:
    outputs.reload()
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"
    command = [
        "python",
        "-m",
        "scripts.build_benign_recovery_data",
        "--out_path",
        remote_out_path,
        "--limit_per_split",
        str(limit_per_split),
        "--seed",
        str(seed),
        "--generator_model",
        generator_model,
        "--judge_model",
        judge_model,
    ]

    for path in eval_paths:
        remote_eval_path = path if path.startswith("/") else f"{OUTPUT_DIR}/{path}"
        command.extend(["--eval_path", remote_eval_path])

    if max_examples is not None:
        command.extend(["--max_examples", str(max_examples)])

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        outputs.commit()

    return remote_out_path


@app.function(
    image=image,
    timeout=6 * 60 * 60,
    secrets=[local_env_secret],
    volumes={
        OUTPUT_DIR: outputs,
    },
)
def run_build_toxicchat_classifier_data(
    out_path: str = f"{OUTPUT_DIR}/hard_training/v4_toxicchat_teacher.jsonl",
    dataset_config: str = "toxicchat0124",
    split: str = "train",
    seed: int = 0,
    max_safe_compliance: int = 300,
    max_judged_model_outputs: int = 150,
    generator_model: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
) -> str:
    outputs.reload()
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"
    command = [
        "python",
        "-m",
        "scripts.build_toxicchat_classifier_data",
        "--out_path",
        remote_out_path,
        "--dataset_config",
        dataset_config,
        "--split",
        split,
        "--seed",
        str(seed),
        "--max_safe_compliance",
        str(max_safe_compliance),
        "--max_judged_model_outputs",
        str(max_judged_model_outputs),
        "--generator_model",
        generator_model,
        "--judge_model",
        judge_model,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        outputs.commit()

    return remote_out_path


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def train_probe(
    model_id: str = DEFAULT_MODEL_ID,
    data_path: str = DEFAULT_DATA_PATH,
    layer: int = -4,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    max_examples: int = 20000,
    no_chat_template: bool = False,
    artifact_id: str | None = None,
) -> dict[str, object]:
    os.environ["HF_HOME"] = HF_CACHE_DIR

    remote_data_path = data_path if data_path.startswith("/") else f"{APP_DIR}/{data_path}"
    remote_out_dir = f"{OUTPUT_DIR}/{out_dir}"

    command = [
        "python",
        "-m",
        "scripts.train_probe",
        "--model_id",
        model_id,
        "--data_path",
        remote_data_path,
        "--layer",
        str(layer),
        "--out_dir",
        remote_out_dir,
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--max_examples",
        str(max_examples),
    ]

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        subprocess.run(command, cwd=APP_DIR, check=True)
    finally:
        hf_cache.commit()
        outputs.commit()

    config_path = Path(remote_out_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "artifact_id": artifact_id or Path(out_dir).name,
        "artifact_type": "probe",
        "artifact_dir": out_dir.strip("/"),
        "volume_path": remote_out_dir,
        "files": ["probe.pt", "config.json"],
        "metrics": config.get("metrics"),
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_train_exchange_classifier(
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    data_path: str = DEFAULT_CLASSIFIER_DATA_PATH,
    out_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    prefix_augment: bool = True,
    prefix_copies: int = 1,
    logging_steps: int = 500,
    save_steps: int = 5000,
    resume_from_checkpoint: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, object]:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    outputs.reload()

    remote_data_path = data_path if data_path.startswith("/") else f"{APP_DIR}/{data_path}"
    remote_out_dir = f"{OUTPUT_DIR}/{out_dir}"
    remote_out_path = Path(remote_out_dir)
    remote_out_path.mkdir(parents=True, exist_ok=True)
    print(f"Classifier training data path: {remote_data_path}")
    print(f"Classifier training output dir: {remote_out_dir}")
    log_path = remote_out_path / "train.log"

    command = [
        "python",
        "-u",
        "-m",
        "scripts.train_exchange_classifier",
        "--model_id",
        model_id,
        "--data_path",
        remote_data_path,
        "--output-dir",
        remote_out_dir,
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--max_length",
        str(max_length),
        "--logging_steps",
        str(logging_steps),
        "--save_steps",
        str(save_steps),
        "--disable_tqdm",
    ]

    if prefix_augment:
        command.append("--prefix_augment")
        command.extend(["--prefix_copies", str(prefix_copies)])

    if resume_from_checkpoint is not None:
        checkpoint_path = _resolve_resume_checkpoint(
            resume_from_checkpoint,
            remote_out_path=remote_out_path,
        )
        command.extend(["--resume_from_checkpoint", str(checkpoint_path)])

    try:
        print(f"Training classifier. Logs are being written to: {log_path}")
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                cwd=APP_DIR,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        print(f"Classifier training completed. Logs: {log_path}")
    except subprocess.CalledProcessError:
        print(f"Classifier training failed. Last lines from {log_path}:")
        print(_read_last_lines(log_path))
        raise
    finally:
        hf_cache.commit()
        outputs.commit()

    classifier_config_path = remote_out_path / "classifier_config.json"
    classifier_config = json.loads(classifier_config_path.read_text(encoding="utf-8"))
    return {
        "artifact_id": artifact_id or Path(out_dir).name,
        "artifact_type": "classifier",
        "artifact_dir": out_dir.strip("/"),
        "volume_path": remote_out_dir,
        "files": [
            "config.json",
            "classifier_config.json",
            "model.safetensors",
            "tokenizer.json",
        ],
        "metrics": classifier_config.get("metrics"),
    }


@app.function(
    image=publish_image,
    timeout=60 * 60,
    volumes={
        OUTPUT_DIR: outputs,
    },
)
def publish_artifact_to_hf(
    artifact_id: str,
    artifact_type: str,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Upload trained Open Constitution artifact",
) -> dict[str, str]:
    """Publish an API-created artifact using HF_TOKEN injected as a Modal Secret."""
    from activation_probe_mvp.publishing import publish_artifact

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN must be supplied through a Modal Secret")

    outputs.reload()
    artifact_dir = _api_artifact_dir(artifact_id, artifact_type)
    result = publish_artifact(
        artifact_dir=artifact_dir,
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        repo_id=repo_id,
        hf_token=hf_token,
        private=private,
        commit_message=commit_message,
    )
    if artifact_type == "probe":
        outputs.commit()
    return result


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_calibrate_exchange_classifier(
    classifier_dir: str,
    data_path: str,
    out_path: str = f"{OUTPUT_DIR}/classifier_calibration.json",
    batch_size: int = 32,
    validation_size: float = 0.1,
    max_records: int | None = 20000,
    max_safe_compliance_block_rate: float = 0.02,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    outputs.reload()

    remote_classifier_dir = (
        classifier_dir if classifier_dir.startswith("/") else f"{OUTPUT_DIR}/{classifier_dir}"
    )
    remote_data_path = data_path if data_path.startswith("/") else f"{APP_DIR}/{data_path}"
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"
    command = [
        "python",
        "-m",
        "scripts.calibrate_exchange_classifier",
        "--classifier_dir",
        remote_classifier_dir,
        "--data_path",
        remote_data_path,
        "--out_path",
        remote_out_path,
        "--batch_size",
        str(batch_size),
        "--validation_size",
        str(validation_size),
        "--max_safe_compliance_block_rate",
        str(max_safe_compliance_block_rate),
    ]
    if max_records is not None:
        command.extend(["--max_records", str(max_records)])

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        raise
    finally:
        hf_cache.commit()
        outputs.commit()

    return remote_out_path


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_guarded_generate(
    prompt: str = DEFAULT_PROMPT,
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str | None = None,
    max_new_tokens: int = 80,
    classifier_block_threshold: float | None = None,
    final_classifier_check: bool = False,
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    outputs.reload()

    remote_probe_dir = f"{OUTPUT_DIR}/{probe_dir}"
    command = [
        "python",
        "-m",
        "scripts.guarded_generate",
        "--probe_path",
        f"{remote_probe_dir}/probe.pt",
        "--config_path",
        f"{remote_probe_dir}/config.json",
        "--prompt",
        prompt,
        "--max_new_tokens",
        str(max_new_tokens),
    ]

    if model_id is not None:
        command.extend(["--model_id", model_id])

    if classifier_dir is not None:
        remote_classifier_dir = f"{OUTPUT_DIR}/{classifier_dir}"
        command.extend(["--classifier_dir", remote_classifier_dir])

    if classifier_block_threshold is not None:
        command.extend(["--classifier_block_threshold", str(classifier_block_threshold)])

    if final_classifier_check:
        command.append("--final_classifier_check")

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        subprocess.run(command, cwd=APP_DIR, check=True)
    finally:
        hf_cache.commit()

    return f"Guarded generation completed with probe output directory: {remote_probe_dir}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_benchmark_latency(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    max_new_tokens: int = 64,
    prompts: list[str] | None = None,
    pause_threshold: float | None = None,
    probe_every_n_tokens: int | None = None,
    classifier_block_threshold: float | None = None,
    final_classifier_check: bool = False,
    out_path: str = f"{OUTPUT_DIR}/latency_benchmark.json",
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    outputs.reload()

    remote_probe_dir = f"{OUTPUT_DIR}/{probe_dir}"
    remote_classifier_dir = f"{OUTPUT_DIR}/{classifier_dir}"
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"
    command = [
        "python",
        "-m",
        "scripts.benchmark_latency",
        "--probe_path",
        f"{remote_probe_dir}/probe.pt",
        "--config_path",
        f"{remote_probe_dir}/config.json",
        "--classifier_dir",
        remote_classifier_dir,
        "--max_new_tokens",
        str(max_new_tokens),
        "--out_path",
        remote_out_path,
    ]

    if model_id is not None:
        command.extend(["--model_id", model_id])

    if prompts:
        for prompt in prompts:
            command.extend(["--prompt", prompt])

    if pause_threshold is not None:
        command.extend(["--pause_threshold", str(pause_threshold)])

    if probe_every_n_tokens is not None:
        command.extend(["--probe_every_n_tokens", str(probe_every_n_tokens)])

    if classifier_block_threshold is not None:
        command.extend(["--classifier_block_threshold", str(classifier_block_threshold)])

    if final_classifier_check:
        command.append("--final_classifier_check")

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        hf_cache.commit()
        outputs.commit()

    return remote_out_path


@app.function(
    image=image,
    gpu="A10G",
    memory=32768,
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret"), local_env_secret],
    volumes={
        "/cache": hf_cache,
        OUTPUT_DIR: outputs,
    },
)
def run_benchmark_jailbreak(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    dataset: str = "behaviors",
    max_new_tokens: int = 64,
    limit_per_split: int = 10,
    max_examples: int | None = None,
    example_offset: int = 0,
    only_split: str | None = None,
    example_indices: list[int] | None = None,
    seed: int = 0,
    pause_threshold: float | None = None,
    probe_every_n_tokens: int | None = None,
    guard_classifier_block_threshold: float | None = None,
    judge_type: str = "classifier",
    openrouter_model: str = "openai/gpt-4o-mini",
    final_classifier_check: bool = False,
    out_path: str = f"{OUTPUT_DIR}/jailbreak_benchmark.json",
    teacher_data_path: str | None = None,
    no_chat_template: bool = False,
) -> str:
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    outputs.reload()

    remote_probe_dir = f"{OUTPUT_DIR}/{probe_dir}"
    remote_classifier_dir = f"{OUTPUT_DIR}/{classifier_dir}"
    remote_out_path = out_path if out_path.startswith("/") else f"{OUTPUT_DIR}/{out_path}"
    command = [
        "python",
        "-m",
        "scripts.benchmark_jailbreak",
        "--probe_path",
        f"{remote_probe_dir}/probe.pt",
        "--config_path",
        f"{remote_probe_dir}/config.json",
        "--classifier_dir",
        remote_classifier_dir,
        "--dataset",
        dataset,
        "--max_new_tokens",
        str(max_new_tokens),
        "--limit_per_split",
        str(limit_per_split),
        "--out_path",
        remote_out_path,
        "--judge_type",
        judge_type,
        "--seed",
        str(seed),
    ]

    if judge_type == "openrouter":
        command.extend(["--openrouter_model", openrouter_model])

    if final_classifier_check:
        command.append("--final_classifier_check")

    if teacher_data_path is not None:
        remote_teacher_data_path = (
            teacher_data_path
            if teacher_data_path.startswith("/")
            else f"{OUTPUT_DIR}/{teacher_data_path}"
        )
        command.extend(["--teacher_data_path", remote_teacher_data_path])

    if max_examples is not None:
        command.extend(["--max_examples", str(max_examples)])

    if example_offset:
        command.extend(["--example_offset", str(example_offset)])

    if only_split is not None:
        command.extend(["--only_split", only_split])

    for example_index in example_indices or []:
        command.extend(["--example_index", str(example_index)])

    if pause_threshold is not None:
        command.extend(["--pause_threshold", str(pause_threshold)])

    if probe_every_n_tokens is not None:
        command.extend(["--probe_every_n_tokens", str(probe_every_n_tokens)])

    if guard_classifier_block_threshold is not None:
        command.extend(
            ["--guard_classifier_block_threshold", str(guard_classifier_block_threshold)]
        )

    if model_id is not None:
        command.extend(["--model_id", model_id])

    if no_chat_template:
        command.append("--no_chat_template")

    try:
        completed = subprocess.run(
            command,
            cwd=APP_DIR,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    finally:
        hf_cache.commit()
        outputs.commit()

    return remote_out_path


@app.local_entrypoint()
def main(
    model_id: str = DEFAULT_MODEL_ID,
    data_path: str = DEFAULT_DATA_PATH,
    layer: int = -4,
    out_dir: str = DEFAULT_OUT_DIR,
    epochs: int = 100,
    lr: float = 1e-3,
    max_examples: int = 20000,
    no_chat_template: bool = False,
):
    remote_out_dir = _volume_path(out_dir)
    function_call = train_probe.spawn(
        model_id=model_id,
        data_path=data_path,
        layer=layer,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        max_examples=max_examples,
        no_chat_template=no_chat_template,
    )
    _print_spawned("probe training", function_call, output_path=remote_out_dir)


@app.local_entrypoint()
def build_three_class_classifier_data(
    primary_input_path: str = "data/training_data_classifier.jsonl",
    extra_input_path: str | None = None,
    out_path: str = "training_data_classifier_threeclass.jsonl",
    keep_extra_duplicates: bool = False,
    extra_repeat: int = 1,
):
    remote_out_path = _volume_path(out_path)
    function_call = run_build_three_class_classifier_data.spawn(
        primary_input_path=primary_input_path,
        extra_input_paths=extra_input_path.split(",") if extra_input_path else None,
        out_path=out_path,
        keep_extra_duplicates=keep_extra_duplicates,
        extra_repeat=extra_repeat,
    )
    _print_spawned("three-class data build", function_call, output_path=remote_out_path)


@app.local_entrypoint()
def build_benign_recovery_data(
    eval_path: str,
    out_path: str = "hard_training/v3_benign_recovery.jsonl",
    limit_per_split: int = 100,
    seed: int = 0,
    generator_model: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
    max_examples: int | None = None,
):
    remote_out_path = _volume_path(out_path)
    function_call = run_build_benign_recovery_data.spawn(
        eval_paths=eval_path.split(","),
        out_path=out_path,
        limit_per_split=limit_per_split,
        seed=seed,
        generator_model=generator_model,
        judge_model=judge_model,
        max_examples=max_examples,
    )
    _print_spawned("benign recovery data build", function_call, output_path=remote_out_path)


@app.local_entrypoint()
def build_toxicchat_classifier_data(
    out_path: str = "hard_training/v4_toxicchat_teacher.jsonl",
    dataset_config: str = "toxicchat0124",
    split: str = "train",
    seed: int = 0,
    max_safe_compliance: int = 300,
    max_judged_model_outputs: int = 150,
    generator_model: str = "openai/gpt-4o-mini",
    judge_model: str = "openai/gpt-4o-mini",
):
    remote_out_path = _volume_path(out_path)
    function_call = run_build_toxicchat_classifier_data.spawn(
        out_path=out_path,
        dataset_config=dataset_config,
        split=split,
        seed=seed,
        max_safe_compliance=max_safe_compliance,
        max_judged_model_outputs=max_judged_model_outputs,
        generator_model=generator_model,
        judge_model=judge_model,
    )
    _print_spawned("ToxicChat classifier data build", function_call, output_path=remote_out_path)


@app.local_entrypoint()
def train_classifier(
    model_id: str = DEFAULT_CLASSIFIER_MODEL_ID,
    data_path: str = DEFAULT_CLASSIFIER_DATA_PATH,
    output_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    epochs: float = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    prefix_augment: bool = True,
    prefix_copies: int = 1,
    logging_steps: int = 500,
    save_steps: int = 5000,
    resume_from_checkpoint: str | None = None,
):
    remote_out_dir = _volume_path(output_dir)
    function_call = run_train_exchange_classifier.spawn(
        model_id=model_id,
        data_path=data_path,
        out_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        prefix_augment=prefix_augment,
        prefix_copies=prefix_copies,
        logging_steps=logging_steps,
        save_steps=save_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    _print_spawned("classifier training", function_call, output_path=remote_out_dir)


@app.local_entrypoint()
def generate(
    prompt: str = DEFAULT_PROMPT,
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str | None = None,
    max_new_tokens: int = 80,
    classifier_block_threshold: float | None = None,
    final_classifier_check: bool = False,
    no_chat_template: bool = False,
):
    result = run_guarded_generate.remote(
        prompt=prompt,
        model_id=model_id,
        probe_dir=probe_dir,
        classifier_dir=classifier_dir,
        max_new_tokens=max_new_tokens,
        classifier_block_threshold=classifier_block_threshold,
        final_classifier_check=final_classifier_check,
        no_chat_template=no_chat_template,
    )
    print(result)


@app.local_entrypoint()
def calibrate_classifier(
    classifier_dir: str,
    data_path: str,
    out_path: str = "classifier_calibration.json",
    batch_size: int = 32,
    validation_size: float = 0.1,
    max_records: int | None = 20000,
    max_safe_compliance_block_rate: float = 0.02,
):
    remote_out_path = _volume_path(out_path)
    function_call = run_calibrate_exchange_classifier.spawn(
        classifier_dir=classifier_dir,
        data_path=data_path,
        out_path=out_path,
        batch_size=batch_size,
        validation_size=validation_size,
        max_records=max_records,
        max_safe_compliance_block_rate=max_safe_compliance_block_rate,
    )
    _print_spawned("classifier calibration", function_call, output_path=remote_out_path)


@app.local_entrypoint()
def benchmark_latency(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    max_new_tokens: int = 64,
    pause_threshold: float | None = None,
    probe_every_n_tokens: int | None = None,
    classifier_block_threshold: float | None = None,
    final_classifier_check: bool = False,
    out_path: str = "latency_benchmark.json",
    no_chat_template: bool = False,
):
    remote_out_path = _volume_path(out_path)
    function_call = run_benchmark_latency.spawn(
        model_id=model_id,
        probe_dir=probe_dir,
        classifier_dir=classifier_dir,
        max_new_tokens=max_new_tokens,
        pause_threshold=pause_threshold,
        probe_every_n_tokens=probe_every_n_tokens,
        classifier_block_threshold=classifier_block_threshold,
        final_classifier_check=final_classifier_check,
        out_path=out_path,
        no_chat_template=no_chat_template,
    )
    _print_spawned("latency benchmark", function_call, output_path=remote_out_path)


@app.local_entrypoint()
def benchmark_jailbreak(
    model_id: str | None = None,
    probe_dir: str = DEFAULT_OUT_DIR,
    classifier_dir: str = DEFAULT_CLASSIFIER_OUT_DIR,
    dataset: str = "behaviors",
    max_new_tokens: int = 64,
    limit_per_split: int = 10,
    max_examples: int | None = None,
    example_offset: int = 0,
    only_split: str | None = None,
    example_index: str | None = None,
    seed: int = 0,
    pause_threshold: float | None = None,
    probe_every_n_tokens: int | None = None,
    guard_classifier_block_threshold: float | None = None,
    judge_type: str = "classifier",
    openrouter_model: str = "openai/gpt-4o-mini",
    final_classifier_check: bool = False,
    out_path: str = "jailbreak_benchmark.json",
    teacher_data_path: str | None = None,
    no_chat_template: bool = False,
):
    remote_out_path = _volume_path(out_path)
    function_call = run_benchmark_jailbreak.spawn(
        model_id=model_id,
        probe_dir=probe_dir,
        classifier_dir=classifier_dir,
        dataset=dataset,
        max_new_tokens=max_new_tokens,
        limit_per_split=limit_per_split,
        max_examples=max_examples,
        example_offset=example_offset,
        only_split=only_split,
        example_indices=[int(index) for index in example_index.split(",")]
        if example_index
        else None,
        seed=seed,
        pause_threshold=pause_threshold,
        probe_every_n_tokens=probe_every_n_tokens,
        guard_classifier_block_threshold=guard_classifier_block_threshold,
        judge_type=judge_type,
        openrouter_model=openrouter_model,
        final_classifier_check=final_classifier_check,
        out_path=out_path,
        teacher_data_path=teacher_data_path,
        no_chat_template=no_chat_template,
    )
    _print_spawned("jailbreak benchmark", function_call, output_path=remote_out_path)
