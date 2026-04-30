import logging
import pandas as pd
import numpy as np
from ml_training.config import CLEAN_DATA_DIR, MODEL_DIR, ENV_VAR_OUTPUT_SUFFIX
from ml_training.utils import setup_logging
from ml_training.investing_allocation_optimizer.utils import (
    BootstrapPortfolioOptimizer,
)
import cloudpickle
import ml_training


def main():
    logger = logging.getLogger(__name__)

    logger.info("Gathering raw data")
    df = pd.read_csv(CLEAN_DATA_DIR / f"data_{ENV_VAR_OUTPUT_SUFFIX}.csv")

    df.set_index("Date", drop=True, inplace=True)

    logger.info("Fitting model")
    mdl = BootstrapPortfolioOptimizer()
    mdl.fit(df=df)

    logger.info("Saving model")
    with open(MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl", "wb") as f:
        cloudpickle.dump((ml_training, mdl), f)


if __name__ == "__main__":
    logger = setup_logging()
    main()
