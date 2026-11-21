"""Configuration for Indian Infrastructure Arbitration Award Estimation using RAG."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ArbitrationConfig:
    sectors: List[str] = field(
        default_factory=lambda: [
            "Highways & Roads",
            "Railways & Metro",
            "Power & Energy",
            "Ports & Maritime",
            "Aviation & Airports",
            "Water & Urban EPC",
        ]
    )

    entity_types: List[str] = field(
        default_factory=lambda: [
            "CLAIMANT",
            "RESPONDENT",
            "SECTOR",
            "FORUM",
            "STATUTE",
            "CLAIM_AMOUNT_CR",
            "AWARD_AMOUNT_CR",
            "INTEREST_RATE",
            "DELAY_MONTHS",
            "CLAUSE_TYPE",
        ]
    )

    n_cases: int = 60
    n_clusters: int = 6
    rag_top_k: int = 4
    cv_folds: int = 5
    random_seed: int = 42

    # Environment variable names for optional live API execution
    google_api_key_env: str = "GOOGLE_API_KEY"
    google_cse_id_env: str = "GOOGLE_CSE_ID"
    deepseek_api_key_env: str = "DEEPSEEK_API_KEY"
    deepseek_model: str = "deepseek-chat"
