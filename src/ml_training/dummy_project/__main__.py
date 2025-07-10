from ml_training.config import MODEL_DIR, CLEAN_DATA_DIR, LOGS_DIR
import logging
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib




def setup_logging():
    log_file_path = LOGS_DIR / f"{datetime.today()}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(" "message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s -" " %(levelname)s - %(" "message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def main():
    logger = logging.getLogger(__name__)

    logger.info("Gathering raw data")
    df = pd.read_csv(CLEAN_DATA_DIR / "data.csv")

    logger.info("Splitting into X, y")
    X = df.iloc[:,0].to_frame()
    y = df['target']

    logger.info("Fitting model")
    mdl = LogisticRegression()
    mdl.fit(X, y)

    logger.info("Saving model")
    joblib.dump(mdl, MODEL_DIR / "model.pkl")


if __name__ == "__main__":
    logger = setup_logging()
    main()
