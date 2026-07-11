"""Streamlit Cloud entrypoint — delegates to the dashboard app.

Streamlit Community Cloud looks for a `streamlit_app.py` at the repo root.
"""
from src.dashboard.app import main

main()
