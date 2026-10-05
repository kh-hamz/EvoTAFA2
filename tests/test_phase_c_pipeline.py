"""End-to-end Phase C orchestration over signed synthetic Phase B artifacts."""

import contextlib
import io
import shutil

from edgefl.client_data.config import load as load_phase_c
from edgefl.client_data.storage import load_completion, read_json, sha256, write_json, rows
from edgefl.data.config import load as load_phase_b
from edgefl.pipelines import (phase_c_assign, phase_c_local_split,
                              phase_c_preprocessing, phase_c_validate)
from edgefl.pipelines.phase_b_common import stage as phase_b_stage
from edgefl.contracts.phase_b import MANIFEST_FIELDS
from test_phase_c import PhaseCFixture, write_csv


class PhaseCOrchestrationTests(PhaseCFixture):
    def signed_phase_b(self):
        archive = self.root / "archive"
        archive.mkdir()
        selected = archive / "selected.csv"
        shutil.copyfile(self.selected, selected)
        source = archive / "source.csv"
        shutil.copyfile(self.selected, source)
        capture = archive / "source.pcap"
        capture.write_bytes(b"synthetic signed capture")
        configs = self.root / "configs"
        configs.mkdir()
        phase_b_values = {
            "schema_version": "phase-b.v2", "scope": "phase_b", "archive": "archive",
            "source_root": "archive", "selected": {"primary": "archive/selected.csv",
            "smoke": "archive/selected.csv"}, "pairings": {"source.csv": "source.pcap"},
            "label_aliases": {}, "seed": 42, "temporal_block_seconds": 300,
            "purge_seconds": 30, "session_inactivity_seconds": 60,
            "panel_per_class": 512, "output": "out_b",
        }
        write_json(configs / "phase_b.json", phase_b_values)
        phase_b = load_phase_b(configs / "phase_b.json", self.root)
        files = []
        for path in sorted(archive.iterdir()):
            files.append({"path": path.relative_to(self.root).as_posix(),
                          "size": path.stat().st_size, "sha256": sha256(path),
                          "kind": path.suffix.lstrip(".")})
        registry = {"schema_version": "phase-b.v2", "files": files,
                    "pairs": [{"csv": "archive/source.csv", "pcap": "archive/source.pcap",
                               "capture_id": "source"}],
                    "selected": phase_b_values["selected"], "label_aliases": {},
                    "unpaired_csv": []}
        with phase_b_stage(phase_b, "audit", "archive", {}) as run:
            write_json(run.directory / "registry.json", registry)
            audit = run.finish({"files": {}}, {"registry": run.directory / "registry.json"})
        with phase_b_stage(phase_b, "provenance", "primary", {"audit": audit}) as run:
            shutil.copyfile(self.provenance, run.directory / "provenance.csv")
            provenance = run.finish({"observations": 35},
                                    {"provenance": run.directory / "provenance.csv"})
        with phase_b_stage(phase_b, "verify-captures", "primary", {"audit": audit, "provenance": provenance}) as run:
            write_csv(run.directory / "evidence.csv", MANIFEST_FIELDS["evidence"], [])
            evidence = run.finish({}, {"evidence": run.directory / "evidence.csv"})
        with phase_b_stage(phase_b, "group", "primary", {"provenance": provenance, "evidence": evidence}) as run:
            write_csv(run.directory / "groups.csv", MANIFEST_FIELDS["groups"], [])
            groups = run.finish({}, {"groups": run.directory / "groups.csv"})
        with phase_b_stage(phase_b, "global-split", "primary", {"groups": groups},
                           {"protocol": "A"}) as run:
            shutil.copyfile(self.splits, run.directory / "splits.csv")
            shutil.copyfile(self.splits, run.directory / "closed_set.csv")
            split = run.finish({"folds": {"fold": {"feasible": True, "interpretation": "session_time_closed_set",
                               "held_out_captures": [], "closed_set_manifest_classes": ["Backdoor", "Normal"]}}},
                               {"splits": run.directory / "splits.csv", "closed_set": run.directory / "closed_set.csv"})
        with phase_b_stage(phase_b, "global-split", "primary", {"groups": groups}, {"protocol": "B"}) as run:
            shutil.copyfile(self.splits, run.directory / "splits.csv")
            split_b = run.finish({"folds": {"fold": {"feasible": True, "interpretation": "binary_unseen_attack_detection",
                                   "held_out_captures": []}}}, {"splits": run.directory / "splits.csv"})
        panels = {}
        for name, split_path in (("a", split), ("b", split_b)):
            with phase_b_stage(phase_b, "trusted-panel", "primary", {"splits": split_path}) as run:
                panel_rows = [{key: row[key] for key in MANIFEST_FIELDS["panel"]} for row in
                              rows(self.splits)
                              if row["partition"] == "trusted"]
                write_csv(run.directory / "panel.csv", MANIFEST_FIELDS["panel"], panel_rows)
                panels[name] = run.finish({}, {"panel": run.directory / "panel.csv"})
        with phase_b_stage(phase_b, "validate-phase-b", "primary",
                           {"audit": audit, "provenance": provenance, "evidence": evidence, "groups": groups,
                            "splits_a": split, "splits_b": split_b, "panel_a": panels["a"], "panel_b": panels["b"]}) as run:
            validation = run.finish({"status": "PASS", "errors": [], "limitations": []}, {})
        phase_c_values = {
            "schema_version": "phase-c.v2", "scope": "phase_c",
            "phase_b_config": "configs/phase_b.json", "output": "out_c", **self.config,
        }
        write_json(configs / "phase_c.json", phase_c_values)
        return load_phase_c(configs / "phase_c.json", self.root), validation, split, provenance

    def test_all_stages_publish_verified_lineage_and_only_gate_enables_training(self):
        config, phase_b_validation, split, provenance = self.signed_phase_b()
        with contextlib.redirect_stdout(io.StringIO()):
            assignments = phase_c_assign.execute(config, phase_b_validation, split,
                                                 "primary", "fold", "near_iid", "multiclass")
            local = phase_c_local_split.execute(config, assignments, "primary")
            preprocessing = phase_c_preprocessing.execute(config, local, provenance, "primary")
            inputs = {"phase_b_validation": phase_b_validation, "split": split,
                      "assignments": assignments, "local_split": local,
                      "preprocessing": preprocessing, "provenance": provenance,
                      "fold": "fold", "scenario": "near_iid"}
            gate = phase_c_validate.execute(config, inputs, "primary")
            self.assertEqual(phase_c_assign.execute(config, phase_b_validation, split,
                                                    "primary", "fold", "near_iid", "multiclass"),
                             assignments)
        self.assertFalse(read_json(assignments)["eligible_for_training"])
        metadata = load_completion(config, gate, "validate-phase-c", "primary")
        self.assertTrue(metadata["eligible_for_training"])
        report = read_json(self.root / metadata["artifacts"]["pretraining_report"]["path"])
        self.assertEqual(report["status"], "PASS", report)
        transformer = self.root / read_json(preprocessing)["artifacts"]["transformer"]["path"]
        transformer.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_completion(config, gate, "validate-phase-c", "primary")
