"""Central public metadata for BehaviorScope-X.

Keep user-facing names and release identifiers in one place so docs, GUI
strings, and packaging metadata do not drift apart.
"""

APP_NAME = "BehaviorScope-X"
APP_SLUG = "behaviorscope_x"
APP_VERSION = "3.0.0"
APP_DESCRIPTION = (
    "GUI and reproducible workflows for amortized pose vision in full-video "
    "computational ethology."
)

ANNOTATION_WORKSPACE_NAME = f"{APP_NAME} Annotation Workspace"
DEFAULT_PROJECT_NAME = f"{APP_NAME} Annotation Project"

# Keep the local database/state folder compatible with existing workspaces.
LOCAL_STATE_DIR = f".{APP_SLUG}"
