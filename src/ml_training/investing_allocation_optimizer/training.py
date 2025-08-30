import logging
import pandas as pd
import numpy as np
import joblib
from ml_training.config import CLEAN_DATA_DIR, MODEL_DIR, ENV_VAR_OUTPUT_SUFFIX
from ml_training.utils import setup_logging
from ml_training.investing_allocation_optimizer.utils import (
    BootstrapPortfolioOptimizer,
)


def main():
    logger = logging.getLogger(__name__)

    logger.info("Gathering raw data")
    df = pd.read_csv(CLEAN_DATA_DIR / f"data_{ENV_VAR_OUTPUT_SUFFIX}.csv")

    logger.info("Fitting model")
    mdl = BootstrapPortfolioOptimizer(
        metric=lambda x: np.mean(x),
    )
    mdl.fit(df)

    logger.info("Saving model")
    joblib.dump(mdl, MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl")


if __name__ == "__main__":
    logger = setup_logging()
    main()
