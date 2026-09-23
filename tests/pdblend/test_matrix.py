"""CPU tests for the matrix runner helpers."""
import json
import pytest

from pdblend.bench.matrix import layout_capacity, parse_kv, run_matrix
from synthetic import synthetic_model


def test_parse_kv():
    assert parse_kv("P=1,D=2,M=1") == {"P": 1, "D": 2, "M": 1}
    assert parse_kv("") == {}


def test_layout_capacity_monotone(tmp_path):
    prof = tmp_path / "p.json"
    synthetic_model().save(prof)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = [dict(prompt="x", input_tokens=300 + 10 * i, output_tokens=200) for i in range(50)]
    (corpus / "sharegpt.json").write_text(json.dumps(dict(evaluation=rows)))
    c4 = layout_capacity(prof, corpus, "sharegpt", "M=4", "M=2520", 0)
    c2 = layout_capacity(prof, corpus, "sharegpt", "M=2", "M=2520", 0)
    c4_low = layout_capacity(prof, corpus, "sharegpt", "M=4", "M=900", 0)
    assert c4 > c2 > 0 and c4 > c4_low


def test_run_matrix_dry_and_skip(tmp_path):
    root = tmp_path / "m3"
    (root / "done").mkdir(parents=True)
    (root / "done" / "summary.json").write_text("{}")
    spec = tmp_path / "m3.json"
    spec.write_text(json.dumps(dict(root=str(root), defaults=dict(policy="manual", rate=2.0),
                                    points=[dict(name="done"), dict(name="todo", layout="M=4")])))
    with pytest.raises(RuntimeError, match="without evidence"):
        run_matrix(spec, dry=True)
    (root / "done" / "evidence.json").write_text(json.dumps({"status": "complete", "returncode": 0}))
    rows = run_matrix(spec, dry=True)
    assert [r["status"] for r in rows] == ["skipped", "dry"]
    assert rows[1]["args"]["layout"] == "M=4" and rows[1]["args"]["rate"] == 2.0
    assert json.loads((root / "matrix-status.json").read_text())[1]["name"] == "todo"


def test_run_matrix_shard_and_gpus(tmp_path):
    root = tmp_path / "m3"
    spec = tmp_path / "m3.json"
    spec.write_text(json.dumps(dict(root=str(root), defaults=dict(policy="manual", gpus="4,5,6,7"),
                                    points=[dict(name=f"p{i}") for i in range(5)])))
    rows = run_matrix(spec, dry=True, shard="1/2", gpus="0,1,2,3")
    assert [r["name"] for r in rows] == ["p1", "p3"]
    assert all(r["args"]["gpus"] == "0,1,2,3" for r in rows)
    assert (root / "matrix-status-1of2.json").exists()


def test_auto_clocks_and_m3_spec(tmp_path):
    from pdblend.bench.matrix import auto_clocks, m3_spec
    prof = tmp_path / "p.json"
    synthetic_model().save(prof)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for ds in ("alpaca", "sharegpt", "longbench"):
        rows = [dict(prompt="x", input_tokens=300 + 10 * i, output_tokens=200) for i in range(50)]
        (corpus / f"{ds}.json").write_text(json.dumps(dict(evaluation=rows)))
    low = auto_clocks(prof, corpus, "sharegpt", "M=4", 1.0, 0)
    high = auto_clocks(prof, corpus, "sharegpt", "M=4", 40.0, 0)
    assert low["M"] <= high["M"] and set(low) == {"P", "D", "M"}
    spec = m3_spec(prof, "4,5,6,7", tmp_path / "m3", corpus=corpus, scales=(0.5, 1.0), duration=30)
    assert len(spec["points"]) == 3 * 2 * 3 * 2 and (tmp_path / "m3" / "spec.json").exists()
    names = [p["name"] for p in spec["points"]]
    assert "sharegpt-x0.5-1P+3D-auto" in names and all(p["rate"] > 0 for p in spec["points"])


def test_eval_spec(tmp_path):
    from pdblend.bench.matrix import ABLATIONS, CORE_POLICIES, PORTED_POLICIES, eval_spec
    from pdblend.control.policies import POLICIES
    prof = tmp_path / "p.json"
    synthetic_model().save(prof)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for ds in ("alpaca", "sharegpt", "longbench"):
        rows = [dict(prompt="x", input_tokens=300 + 10 * i, output_tokens=200) for i in range(50)]
        (corpus / f"{ds}.json").write_text(json.dumps(dict(evaluation=rows)))
    spec = eval_spec(prof, "0,1,2,3,4,5,6,7", tmp_path / "eval", corpus=corpus, scales=(0.5,), duration=60,
                     stages="60:0.5,60:1.0", azure=("conv",), azure_duration=120)
    assert (tmp_path / "eval" / "spec.json").exists()
    assert spec["groups"] == dict(controlled=3 * len(CORE_POLICIES), ported=2 * len(PORTED_POLICIES),
                                  ablation=2 * len(ABLATIONS), staged=2 * len(CORE_POLICIES), azure=2 * 3)
    assert len(spec["points"]) == sum(spec["groups"].values())
    assert all(p["policy"] in POLICIES for p in spec["points"])
    assert len({p["name"] for p in spec["points"]}) == len(spec["points"])
    staged = next(p for p in spec["points"] if p["name"] == "sharegpt-staged-pdblend")
    assert staged["duration"] == 120 and staged["rate"] == spec["capacity_rps"]["sharegpt"]
    az = next(p for p in spec["points"] if p["name"] == "longbench-azure-conv-pdblend")
    assert az["azure"] == "conv" and az["duration"] == 120 and 0 < az["azure_peak"] < spec["capacity_rps"]["longbench"]
