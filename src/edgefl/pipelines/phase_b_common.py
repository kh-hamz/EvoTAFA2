"""Stage lifecycle: explicit dependencies, immutable attempts, events, and completion last."""

import hashlib
import json
import platform
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from edgefl.config import canonical_json, workspace_path
from edgefl.data.storage import (VERSION, checked, implementation_hash, load_completion,
                                read_json, ref, write_json)


class Stage:
    def __init__(self, config, name, dataset, inputs, options=None):
        self.config, self.name, self.dataset = config, name, dataset
        self.inputs = {key:ref(config.workspace,path,"foundation" if key=="foundation" else "completion")
                       for key,path in inputs.items()}
        self.fingerprint = implementation_hash(name)
        identity = hashlib.sha256(canonical_json([config.sha256,self.fingerprint,name,dataset,
                                                  self.inputs,options or {}]).encode()).hexdigest()
        base = workspace_path(config.workspace,config.values["output"]) / config.sha256[:16] / dataset / (name+"-"+identity[:16])
        self.reused = None
        if base.exists():
            for completion in sorted(base.glob("*/completion.json")):
                metadata = load_completion(config.workspace,completion,name,config.sha256)
                if metadata["inputs"] == self.inputs and metadata["options"] == (options or {}):
                    self.reused = completion
                    self.directory = completion.parent
                    return
        self.directory = base / ("attempt-" + uuid.uuid4().hex[:12])
        self.directory.mkdir(parents=True,exist_ok=False)
        self.options = options or {}
        self.started = time.monotonic()
        self.event(event="started")
        write_json(self.directory / "configuration.json",config.values)
        write_json(self.directory / "environment.json",
                   {"python":platform.python_version(),"platform":platform.platform(),"sqlite":sqlite3.sqlite_version})

    def event(self, **value):
        entry = {"stage":self.name,"elapsed_seconds":round(time.monotonic()-self.started,3),**value}
        with (self.directory / "events.jsonl").open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(canonical_json(entry)+"\n")
        print(canonical_json(entry),flush=True)

    def finish(self, summary, artifacts):
        write_json(self.directory / "summary.json",summary)
        self.event(event="completed")
        paths = {**artifacts,"summary":self.directory/"summary.json",
                 "configuration":self.directory/"configuration.json",
                 "environment":self.directory/"environment.json","events":self.directory/"events.jsonl"}
        metadata = {"schema_version":VERSION,"status":"complete","stage":self.name,"dataset":self.dataset,
                    "configuration_sha256":self.config.sha256,"implementation_sha256":self.fingerprint,
                    "inputs":self.inputs,"options":self.options,
                    "artifacts":{key:ref(self.config.workspace,path,key) for key,path in paths.items()},
                    "eligible_for_training":False}
        write_json(self.directory / "completion.json",metadata)
        return self.directory / "completion.json"


@contextmanager
def stage(config,name,dataset,inputs,options=None):
    run = Stage(config,name,dataset,inputs,options)
    try:
        yield run
    except BaseException as exc:
        if not run.reused:
            run.event(event="failed",error=type(exc).__name__)
            write_json(run.directory/"failure.json",{"status":"incomplete","error":str(exc),
                        "exception":type(exc).__name__,"eligible_for_training":False})
        raise


def upstream(config,path,expected,dataset=None):
    metadata = load_completion(config.workspace,path,expected,config.sha256)
    if dataset and metadata["dataset"] != dataset:
        raise ValueError("Upstream selected dataset mismatch")
    return metadata


def artifact(config,metadata,name):
    return workspace_path(config.workspace,metadata["artifacts"][name]["path"])


def foundation(root,path):
    from edgefl.data.storage import load_foundation
    return load_foundation(root,path)


def require_link(config,metadata,key,path):
    if metadata["inputs"].get(key) != ref(config.workspace,path,"completion"):
        raise ValueError(f"Upstream lineage mismatch for {key}")


def disk_check(root,registry):
    # Conservative upper bound includes portable manifests, packet spools, and SQLite.
    source_bytes = sum(entry["size"] for entry in registry["files"])
    estimate = max(128*1024**2, source_bytes*4)
    free = shutil.disk_usage(root).free
    if free < estimate:
        raise ValueError(f"Insufficient disk: estimated {estimate} bytes, available {free}")
    return {"source_bytes":source_bytes,"estimated_working_bytes":estimate,"free_bytes":free}
