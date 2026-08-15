# Circular Import Fix

## Root cause

Streamlit executed `streamlit_app/app.py` as module name `app`, which shadowed
the real `app` package. When `streamlit_app/bootstrap.py` imported
`from app.container import ...`, Python resolved the partially initialized
Streamlit script instead of the application package.

## Fix

- Renamed entry module to `streamlit_app/main.py`
- Split adapter into `streamlit_app/backend.py` and `streamlit_app/startup.py`
- Kept `streamlit_app/bootstrap.py` as a thin re-export only
- Added `streamlit_app/runtime.py` to configure `sys.path` before backend imports

## Measured validation

- Import chain: **PASS** — StreamlitBackend import chain OK
- No app shadowing: **PASS** — Streamlit entry loads without shadowing app package
