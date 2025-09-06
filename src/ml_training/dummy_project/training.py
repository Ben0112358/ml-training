import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ml_training.config import CLEAN_DATA_DIR, MODEL_DIR, ENV_VAR_OUTPUT_SUFFIX
from ml_training.utils import setup_logging
import cloudpickle


def main():
    logger = logging.getLogger(__name__)

    logger.info("Gathering raw data")
    df = pd.read_csv(CLEAN_DATA_DIR / f"data_{ENV_VAR_OUTPUT_SUFFIX}.csv")

    logger.info("Splitting into X, y")
    X = df.iloc[:, 0].to_frame()
    y = df["target"]

    logger.info("Fitting model")
    mdl = LogisticRegression()
    mdl.fit(X, y)

    logger.info("Saving model")
    with open(MODEL_DIR / f"model_{ENV_VAR_OUTPUT_SUFFIX}.pkl", "wb") as f:
        cloudpickle.dump(mdl, f)


if __name__ == "__main__":
    logger = setup_logging()
    main()
