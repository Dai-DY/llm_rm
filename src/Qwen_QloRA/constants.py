LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_COLUMNS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

DEFAULT_MODEL_PATH = "models/Qwen__Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT_DIR = "output/Qwen_QloRA"

