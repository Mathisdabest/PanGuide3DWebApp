import sys, types
import pytest
import gradio as gr
import app

def test_consent_gate():
    with pytest.raises(gr.Error): app.run_scan('unused.nii',False)

def test_gpu_failure_never_returns_predictions(monkeypatch):
    monkeypatch.setattr(app,'load_upload',lambda _: (None,None))
    def fail(_): raise RuntimeError('simulated unavailable GPU')
    monkeypatch.setattr(app,'infer',fail)
    app.RATE.clear()
    with pytest.raises(gr.Error,match='did not complete'):
        app.run_scan('fixture.nii',True)

def test_rate_limit():
    app.RATE.clear()
    for _ in range(3): app.rate_limit(None)
    with pytest.raises(gr.Error,match='wait'): app.rate_limit(None)
    app.RATE.clear()

def test_no_cache_is_not_faked(monkeypatch,tmp_path):
    monkeypatch.setattr(app,'CACHE',tmp_path)
    with pytest.raises(gr.Error,match='No real cached masks'):
        app.sample('missing',True,True)
