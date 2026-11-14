# Registers empty stand-ins for torch / open_clip / PIL when they aren't installed (see fakes.py), so modules that
# import app.sessions can be collected on a machine without the CLIP stack. Installed packages are left alone.
import fakes  # noqa: F401
