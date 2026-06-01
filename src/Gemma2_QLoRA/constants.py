LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_COLUMNS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

DEFAULT_MODEL_PATH = "models/sfairXC__FsfairX-Gemma2-RM-v0.1"
DEFAULT_OUTPUT_DIR = "output/gemma2_qlora_rm"

