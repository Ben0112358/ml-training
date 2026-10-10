import json
import logging

import cloudpickle
import ml_training
from ml_training.board_meeting_protocol.utils import ProtocolWriter
from ml_training.config import (
    CLEAN_DATA_DIR,
    ENV_VAR_OUTPUT_SUFFIX,
    MODEL_DIR,
)
from ml_training.utils import setup_logging


def main() -> None:
    logger = logging.getLogger(__name__)
    marker_path = CLEAN_DATA_DIR / f"data_{ENV_VAR_OUTPUT_SUFFIX}.json"
    logger.info("Reading data marker %s", marker_path.name)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("corpus") != "none":
        raise RuntimeError("Expected a corpus-none marker from ml-data")

    logger.info("Packing prompt briefs")
    writer = ProtocolWriter().fit()
    with open(
        MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl", "wb"
    ) as handle:
        cloudpickle.dump((ml_training, writer), handle)


if __name__ == "__main__":
    setup_logging()
    main()
