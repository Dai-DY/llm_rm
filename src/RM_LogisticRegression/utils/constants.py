LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]

METHOD_NAME = "RM_LogisticRegression"

RM_BASE_FEATURE_COLUMNS = [
    "score_a",
    "score_b",
    "score_diff",
    "score_abs_diff",
    "prompt_len",
    "response_a_len",
    "response_b_len",
    "response_len_diff",
]

RM_DERIVED_FEATURE_COLUMNS = [
    "score_sum",
    "score_mean",
    "score_product",
    "score_max",
    "score_min",
    "response_len_abs_diff",
    "response_len_sum",
    "response_len_mean",
    "response_len_ratio",
    "log_prompt_len",
    "log_response_a_len",
    "log_response_b_len",
]

RM_FEATURE_COLUMNS = [*RM_BASE_FEATURE_COLUMNS, *RM_DERIVED_FEATURE_COLUMNS]
