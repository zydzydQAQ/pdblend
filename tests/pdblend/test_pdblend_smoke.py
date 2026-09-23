import json


def test_smoke_wrapper_records_seed_and_nonformal_scope(monkeypatch, tmp_path):
    from pdblend.bench import pdblend_smoke

    corpus = tmp_path / "sharegpt.json"
    corpus.write_text(json.dumps({"evaluation": [
        {"prompt": ["hello"], "output_tokens": 16},
        {"prompt": ["world"], "output_tokens": 16},
    ]}))

    def fake_run_point(*args, **kwargs):
        out = args[7]
        trace = args[5]
        (out / "outcomes.jsonl").write_text("".join(
            json.dumps({"idx": r.idx, "error": None, "completion_tokens": 16,
                        "first_token_s": 1.0, "finished_s": 2.0, "path": "M",
                        "sampling_seed": 701}) + "\n"
            for r in trace))
        (out / "controller.jsonl").write_text('{}\n')
        (out / "power.jsonl").write_text('[0, [1.0]]\n')
        (out / "metering.json").write_text('{"error": null}\n')
        return {"requests": 1}

    monkeypatch.setattr(pdblend_smoke, "run_point", fake_run_point)
    result = pdblend_smoke.run_smoke(
        model="Qwen2.5-7B-Instruct", gpus=[0, 1], tp=1,
        profile=tmp_path / "profile.json", corpus=tmp_path,
        out=tmp_path / "out", duration=100.0)
    assert result["status"] == "passed"
    assert result["complete"] is True
    assert result["seed"] == 701
    assert result["formal_eligible"] is False
    assert result["energy_comparable"] is False
    assert result["scope"] == {"dynamic_tp": False, "complete_kv": False}
    assert (tmp_path / "out/completion.json").is_file()
