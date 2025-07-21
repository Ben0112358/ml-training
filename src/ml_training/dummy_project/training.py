import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from ml_training.config import CLEAN_DATA_DIR, MODEL_DIR
from ml_training.utils import setup_logging


def main():
    logger = logging.getLogger(__name__)

    logger.info("Gathering raw data")
    df = pd.read_csv(CLEAN_DATA_DIR / "data.csv")

    logger.info("Splitting into X, y")
    X = df.iloc[:, 0].to_frame()
    y = df["target"]

    logger.info("Fitting model")
    mdl = LogisticRegression()
    mdl.fit(X, y)

    logger.info("Saving model")
    joblib.dump(mdl, MODEL_DIR / "model.pkl")


if __name__ == "__main__":
    logger = setup_logging()
    main()
