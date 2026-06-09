"""Central public metadata for BehaviorScope-X.

Keep user-facing names, compatibility slugs, and release identifiers in one
place so docs, GUI strings, and packaging metadata do not drift apart.
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

TUTORIAL_FOLDER_NAME = "BehaviorScope-Y_tutorial"
TUTORIAL_OUTPUTS_DIR = "BehaviorScope-Y_tutorial_outputs"
DEFAULT_TUTORIAL_HF_REPO = "farhanaugustine/BehaviorScope-Y_tutorial"

# Some tutorial and schema assets were prepared under the BehaviorScope-Y
# working name. Keep those identifiers stable for saved models and manifests.
LEGACY_APP_NAME = "BehaviorScope-Y"
LEGACY_APP_SLUG = "behaviorscope_y"

# Keep the local database/state folder compatible with existing workspaces.
LOCAL_STATE_DIR = f".{APP_SLUG}"
