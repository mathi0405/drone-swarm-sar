"""Hugging Face Spaces entry point — thin wrapper over the packaged dashboard.

Keeping the dashboard logic in the installed package (swarm_sar.dashboard.app)
means the Space and the local `streamlit run src/swarm_sar/dashboard/app.py`
stay identical; this file only exists because Spaces expects a top-level
app.py.
"""
import runpy

runpy.run_module("swarm_sar.dashboard.app", run_name="__main__")
