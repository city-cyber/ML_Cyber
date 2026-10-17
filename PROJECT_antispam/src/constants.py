"""Общие параметры воспроизводимого эксперимента."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent
DATASETS = {"easy_ham": 0, "hard_ham": 0, "spam_2": 1}
RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TARGET_PRECISION = 0.99
MAX_FPR = 0.01
EXTERNAL_URL = "https://disk.yandex.ru/d/mJ_9zG3dX3eZgg"
EXTERNAL_SHA256 = "14ccf52ab11673a85acab7c9793effa2c8db72c61838ba1b15356bd22d9576ee"
# В старой версии spam_2 это обычное письмо ошибочно размечено как спам.
# Исправление описано автором корпуса в истории изменений от 11.03.2005.
MISLABELED_SPAM = "00529.0c8a07bb7b14576063ba0c1c4079e209"
