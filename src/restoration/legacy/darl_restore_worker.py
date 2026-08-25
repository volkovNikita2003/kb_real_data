from __future__ import annotations

import traceback
from pathlib import Path


def run_darl_restore_worker(
    connection,
    classes,
    restored_distr,
    config_name,
    dir_save,
    prefix,
):
    """Calculate the DARL signal in an isolated process."""
    try:
        # Keep DARL and its native dependencies out of the main process.
        from restoration.legacy.darl_restore import get_signal_from_restore

        b = get_signal_from_restore(
            classes=classes,
            restored_distr=restored_distr,
            config_name=config_name,
            dir_save=Path(dir_save),
            prefix=prefix,
        )
        connection.send(("success", b))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
