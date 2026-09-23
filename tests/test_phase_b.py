"""Focused integrity tests and a tiny complete Phase B pipeline."""

import contextlib
import csv
import io
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from edgefl.config import load_config
from edgefl.data.audit import audit_csv
from edgefl.data.config import load
from edgefl.data.csv_reader import header, records
from edgefl.data.pcap import clock_offset, packet_rows, run_tool
from edgefl.data.schema import canonical, evidence_key, label, projected_time, semantic_issues
from edgefl.data.splitting import allocate
from edgefl.data.storage import checked, load_completion, read_json, ref, rows, sha256, write_json
from edgefl.pipelines import (phase_b_audit, phase_b_provenance, phase_b_captures,
    phase_b_group, phase_b_split, phase_b_panel, phase_b_validate)
from edgefl.pipelines.phase_b_common import artifact, stage
from edgefl.runs import initialize_foundation_run

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, names, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        output=csv.writer(stream,lineterminator="\n")
        output.writerow(names)
        output.writerows(values)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_quoted_delimiters_and_multiline_are_logical_records(self):
        path=self.root/"sample.csv"
        write_csv(path,("a","b"),[("x,y",'a"b\nc'),("d","e")])
        result=list(records(path))
        self.assertEqual(len(result),2)
        self.assertEqual(result[0].values,("x,y",'a"b\nc'))
        self.assertEqual(result[1].number,2)

    def test_width_errors_counted_beyond_example_budget(self):
        path=self.root/"sample.csv"
        write_csv(path,("Attack_type","Attack_label"),[("Normal","0"),*([("bad",)]*25)])
        report=audit_csv(path,self.root/"work.sqlite",{})
        self.assertEqual(report["logical_records"],26)
        self.assertEqual(report["malformed_records"],25)
        self.assertEqual(len(report["diagnostic_examples"]),20)

    def test_duplicate_header_rejected(self):
        path=self.root/"sample.csv"
        path.write_text("x,x\n1,2\n")
        with self.assertRaises(ValueError):
            header(path)

    def test_invalid_encoding_rejected(self):
        path=self.root/"sample.csv"
        path.write_bytes(b"x\n\xff\n")
        with self.assertRaises(ValueError):
            list(records(path))

    def test_unclosed_quote_rejected(self):
        path=self.root/"sample.csv"
        path.write_text('x\n"unclosed\n')
        with self.assertRaises(ValueError):
            list(records(path))

    def test_audit_duplicate_and_chunk_determinism(self):
        path=self.root/"sample.csv"
        write_csv(path,("frame.time","Attack_type","Attack_label"),[("0.0","Normal","0")]*3)
        a=audit_csv(path,self.root/"a.sqlite",{},chunk_size=1)
        b=audit_csv(path,self.root/"b.sqlite",{},chunk_size=100)
        self.assertEqual(a,b)
        self.assertEqual(a["exact_duplicate_occurrences"],2)
        self.assertEqual(a["field_cardinalities"]["Attack_type"],1)


class SemanticTests(unittest.TestCase):
    def test_field_specific_numeric_and_hex_rules(self):
        self.assertEqual(canonical("tcp.checksum","0x0010"),canonical("tcp.checksum","16.0"))
        self.assertEqual(canonical("tcp.seq","1e2"),"100")
        self.assertEqual(canonical("tcp.payload","0001"),"0001")
        self.assertEqual(canonical("unknown_field","1.0"),"1.0")

    def test_explicit_alias_and_zero_sentinel(self):
        self.assertEqual(label("OS_Fingerprinting",{"OS_Fingerprinting":"Fingerprinting"}),"Fingerprinting")
        self.assertEqual(canonical("http.file_data","0.0"),"0")
        self.assertEqual(canonical("frame.time","0.0"),"0.0")

    def test_nonfinite_and_misplaced_values(self):
        issues=semantic_issues(("tcp.seq","tcp.srcport","ip.src_host"),("nan","_ipp._tcp.local","7"),{})
        self.assertIn("tcp.seq:nonfinite",issues)
        self.assertIn("tcp.srcport:malformed_numeric",issues)
        self.assertIn("ip.src_host:invalid_address",issues)

    def test_missing_date_is_not_invented(self):
        self.assertIsNone(projected_time("6.0"))
        self.assertIsNone(evidence_key(("frame.time",),("0.0",)))
        self.assertEqual(projected_time("2021 11:00:00.1"),"2021 11:00:00.100000000")

    def test_clock_offset_comes_from_packet_epoch(self):
        epoch=datetime(2021,1,1,10,tzinfo=timezone.utc).timestamp()
        self.assertEqual(clock_offset("2021 11:00:00.000000000",str(epoch)),3600)
        self.assertIsNone(clock_offset("6.0",str(epoch)))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"tool.txt"

    def test_nonzero_exit_checked(self):
        with patch("edgefl.data.pcap.subprocess.run",return_value=subprocess.CompletedProcess([],2)):
            with self.assertRaisesRegex(ValueError,"exited 2"):
                run_tool(["fake"],self.path)

    def test_timeout_checked(self):
        with patch("edgefl.data.pcap.subprocess.run",side_effect=subprocess.TimeoutExpired("fake",1)):
            with self.assertRaisesRegex(ValueError,"TimeoutExpired"):
                run_tool(["fake"],self.path,1)

    def test_missing_executable_checked(self):
        with patch("edgefl.data.pcap.subprocess.run",side_effect=FileNotFoundError):
            with self.assertRaises(ValueError):
                run_tool(["fake"],self.path)

    def test_malformed_packet_output_rejected(self):
        self.path.write_text("frame.number\tframe.time_epoch\n1\tNaN\n")
        with self.assertRaises(ValueError):
            list(packet_rows(self.path))


class AllocationTests(unittest.TestCase):
    def groups(self):
        return [{"group_id":str(i),"start":i*100.0,"end":i*100.0+1,"count":10} for i in range(100)]

    def test_deterministic_and_purged(self):
        groups=self.groups()
        a,b=allocate(groups,(.7,.075,.075,.15),30)
        self.assertEqual((a,b),allocate(list(reversed(groups)),(.7,.075,.075,.15),30))
        self.assertTrue(any(role=="excluded" for role,_ in a.values()))
        self.assertEqual(a["0"][0],"client_pool")
        self.assertEqual(a["99"][0],"final_test")

    def test_long_boundary_crossing_session_excluded(self):
        groups=self.groups()
        groups[0]["end"]=9999
        result,_=allocate(groups,(.7,.075,.075,.15),30)
        self.assertEqual(result["0"],("excluded","boundary_crossing_session_or_group"))

    def test_too_few_groups_is_not_redrawn(self):
        result,boundaries=allocate(self.groups()[:1],(.7,.075,.075,.15),30)
        self.assertLessEqual(len(set(role for role,_ in result.values())),1)
        self.assertEqual(len(boundaries),3)
        self.assertEqual(result["0"][0],"client_pool")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/"configs").mkdir()
        (self.root/"docs").mkdir()
        (self.root/"docs/research_protocol.md").write_text("Fixture protocol")
        self.names=("frame.time","ip.src_host","ip.dst_host","tcp.srcport","tcp.dstport",
                    "tcp.seq","tcp.ack_raw","Attack_label","Attack_type")
        self.packet_data={}
        selected=[]
        base=1609459200
        captures=(("benign1","Normal"),("benign2","Normal"),("attack1","Backdoor"),("attack2","DDoS_TCP"))
        for c,(capture,kind) in enumerate(captures):
            values=[]
            packets=[]
            for n in range(80):
                epoch=base+c*20000+n*100
                stamp=datetime.fromtimestamp(epoch,timezone.utc).strftime("%Y %H:%M:%S")+".000000000"
                row=(stamp,f"192.168.0.{c+1}","192.168.1.1","12345","80",str(n+1),str(n+1000),"0" if kind=="Normal" else "1",kind)
                values.append(row)
                packet=dict(zip(self.names,row))
                packet.update({"frame.number":str(n+1),"frame.time_epoch":str(epoch),
                               "tcp.stream":str(n),"udp.stream":"","ip.proto":"6","tcp.flags":"0x0010",
                               "udp.srcport":"","udp.dstport":""})
                packets.append(packet)
            self.packet_data[capture]=packets
            write_csv(self.root/f"archive/{capture}.csv",self.names,values)
            (self.root/f"archive/{capture}.pcap").write_bytes(b"fixture "+capture.encode())
            selected.extend(values)
        selected.append(selected[0])
        write_csv(self.root/"archive/selected.csv",self.names,selected)
        cfg=json.loads((ROOT/"configs/phase_b.json").read_text())
        cfg.update(archive="archive",source_root="archive",output="out",selected={"primary":"archive/selected.csv","smoke":"archive/selected.csv"},
                   pairings={c+".csv":c+".pcap" for c,_ in captures})
        write_json(self.root/"configs/phase_b.json",cfg)
        self.config=load(self.root/"configs/phase_b.json",self.root)
        a=json.loads((ROOT/"configs/phase_a.json").read_text())
        a["dataset"]={"primary":"archive/selected.csv","smoke":"archive/selected.csv"}
        write_json(self.root/"configs/phase_a.json",a)
        self.foundation=initialize_foundation_run(load_config(self.root/"configs/phase_a.json",self.root),11)/"run.json"
        self.quiet=contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__,None,None,None)

    def fake_extract(self,executable,pcap,names,work,timeout=1800):
        destination=work/"packets.tsv"
        packets=self.packet_data[pcap.stem]
        fields=list(packets[0])
        with destination.open("w",newline="",encoding="utf-8") as stream:
            out=csv.DictWriter(stream,fieldnames=fields,delimiter="\t",lineterminator="\n")
            out.writeheader()
            out.writerows(packets)
        return destination

    def audit_provenance(self):
        audit=phase_b_audit.execute(self.config,self.foundation)
        provenance=phase_b_provenance.execute(self.config,audit,"primary")
        return audit,provenance

    def capture(self,audit,provenance):
        identity={"path":"fake","sha256":"0"*64,"version":"fixture"}
        def summary(arguments,output,timeout=1800):
            output.write_text("fixture capture summary")
        with patch("edgefl.data.pcap.tool_identity",return_value=identity),patch("edgefl.data.pcap.run_tool",side_effect=summary),patch("edgefl.data.pcap.extract",side_effect=self.fake_extract):
            return phase_b_captures.execute(self.config,audit,provenance,"primary")

    def summary(self,completion):
        meta=read_json(completion)
        return read_json(self.root/meta["artifacts"]["summary"]["path"])

    def test_complete_pipeline_and_source_preservation(self):
        before=sha256(self.root/"archive/selected.csv")
        audit,provenance=self.audit_provenance()
        evidence=self.capture(audit,provenance)
        self.assertEqual(self.summary(evidence)["statuses"]["verified"],321)
        groups=phase_b_group.execute(self.config,provenance,evidence,"primary")
        inputs={"audit":audit,"provenance":provenance,"evidence":evidence,"groups":groups}
        for protocol in ("a","b"):
            split=phase_b_split.execute(self.config,groups,"primary",protocol.upper())
            panel=phase_b_panel.execute(self.config,split,"primary")
            inputs["splits_"+protocol]=split
            inputs["panel_"+protocol]=panel
            self.assertTrue(any(v["feasible"] for v in self.summary(split)["folds"].values()))
        final=phase_b_validate.execute(self.config,inputs,"primary")
        report=self.summary(final)
        self.assertEqual(report["status"],"PASS_WITH_LIMITATIONS",report)
        self.assertFalse(report["eligible_for_training"])
        self.assertEqual(report["verified_observations"],320)
        self.assertEqual(before,sha256(self.root/"archive/selected.csv"))
        self.assertEqual(phase_b_audit.execute(self.config,self.foundation),audit)

    def test_equal_counts_without_predictor_correspondence_quarantined(self):
        audit,provenance=self.audit_provenance()
        for packets in self.packet_data.values():
            for packet in packets:
                packet["tcp.ack_raw"]="999999"
        evidence=self.capture(audit,provenance)
        self.assertEqual(self.summary(evidence)["statuses"],{"quarantined":321})

    def test_timestamp_reset_blocks_capture(self):
        audit,provenance=self.audit_provenance()
        self.packet_data["benign1"][10]["frame.time_epoch"]="1609459200"
        evidence=self.capture(audit,provenance)
        report=self.summary(evidence)
        self.assertEqual(report["captures"]["benign1"]["status"],"quarantined")

    def test_packet_sequence_reversal_blocks_capture(self):
        audit,provenance=self.audit_provenance()
        self.packet_data["benign1"][10]["frame.number"]="1"
        evidence=self.capture(audit,provenance)
        self.assertEqual(self.summary(evidence)["captures"]["benign1"]["status"],"quarantined")

    def test_ambiguous_and_unmatched_provenance(self):
        source=self.root/"archive/benign1.csv"
        with source.open("a",encoding="utf-8") as stream:
            stream.write(",".join(next(iter(self.packet_data["benign1"]))[name] for name in self.names)+"\n")
        selected=self.root/"archive/selected.csv"
        with selected.open("a",encoding="utf-8") as stream:
            row=[self.packet_data["benign1"][0][name] for name in self.names]
            row[5]="987654321"
            stream.write(",".join(row)+"\n")
        _,provenance=self.audit_provenance()
        statuses=self.summary(provenance)["statuses"]
        self.assertEqual(statuses["ambiguous"],2)
        self.assertEqual(statuses["unmatched"],1)

    def test_artifact_tampering_rejected(self):
        audit=phase_b_audit.execute(self.config,self.foundation)
        meta=read_json(audit)
        (self.root/meta["artifacts"]["registry"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            load_completion(self.root,audit,"audit",self.config.sha256)

    def test_stale_configuration_and_implementation_rejected(self):
        audit=phase_b_audit.execute(self.config,self.foundation)
        with self.assertRaisesRegex(ValueError,"configuration"):
            load_completion(self.root,audit,"audit","f"*64)
        with patch("edgefl.data.storage.implementation_hash",return_value="f"*64):
            with self.assertRaisesRegex(ValueError,"implementation"):
                load_completion(self.root,audit,"audit",self.config.sha256)

    def test_source_tampering_rejected(self):
        audit=phase_b_audit.execute(self.config,self.foundation)
        with (self.root/"archive/benign1.csv").open("a") as stream:
            stream.write("tampered\n")
        with self.assertRaisesRegex(ValueError,"Registered source changed"):
            phase_b_provenance.execute(self.config,audit,"primary")
        from edgefl.cli import main
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(["--workspace",str(self.root),"audit","--config","configs/phase_b.json",
                           "--foundation",self.foundation.relative_to(self.root).as_posix()])
        self.assertEqual(result,2)  # Cached audit must not bypass registered-source checks.

    def test_partial_stage_never_reused(self):
        with self.assertRaisesRegex(RuntimeError,"interrupted"):
            with stage(self.config,"fixture","primary",{}) as run:
                folder=run.directory
                raise RuntimeError("interrupted")
        self.assertTrue((folder/"failure.json").is_file())
        self.assertFalse((folder/"completion.json").exists())
        with stage(self.config,"fixture","primary",{}) as run:
            self.assertNotEqual(run.directory,folder)

    def test_safe_paths_and_role_check(self):
        config=json.loads(self.config.canonical)
        for unsafe in ("src/new","archive/out","../outside",".git/out"):
            config["output"]=unsafe
            write_json(self.root/"configs/bad.json",config)
            with self.assertRaises(ValueError):
                load(self.root/"configs/bad.json",self.root)
        artifact_ref=ref(self.root,self.foundation,"foundation")
        with self.assertRaisesRegex(ValueError,"role"):
            checked(self.root,artifact_ref,"trusted")

    def test_audit_does_not_execute_later_stages(self):
        with patch("edgefl.data.provenance.recover",side_effect=AssertionError("unexpected")):
            phase_b_audit.execute(self.config,self.foundation)


if __name__=="__main__":
    unittest.main()
