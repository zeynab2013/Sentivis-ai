"""Compatibility exports for the Streamlit backend adapter.

Import ``StreamlitBackend`` from ``streamlit_app.backend`` and
``initialize_backend`` from ``streamlit_app.startup`` in new code.
"""

from streamlit_app.backend import StreamlitBackend
from streamlit_app.startup import initialize_backend

__all__ = ["StreamlitBackend", "initialize_backend"]
