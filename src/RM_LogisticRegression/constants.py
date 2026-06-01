LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]

METHOD_NAME = "RM_LogisticRegression"

RM_FEATURE_COLUMNS = [
    "score_a",
    "score_b",
    "score_diff",
    "score_abs_diff",
    "prompt_len",
    "response_a_len",
    "response_b_len",
    "response_len_diff",
]
