"""HTTP payload schema for one synthetic card-not-present transaction."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(ge=0, le=25_000, description="Synthetic currency units")
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6, description="Synthetic day index 0-6")
    merchant_category: int = Field(
        ge=0,
        le=9,
        description="Synthetic category id. Not a card-network MCC.",
    )
    international: bool
    billing_shipping_mismatch: bool
    txn_count_1h: int = Field(ge=0, le=40)
    txn_count_24h: int = Field(ge=0, le=100)
    mean_amount_7d: float = Field(ge=0, le=10_000)
    device_age_days: int = Field(ge=0, le=5_000)
    account_age_days: int = Field(ge=0, le=10_000)
    km_from_prev: float = Field(ge=0, le=20_000)

    @model_validator(mode="after")
    def velocity_window_is_consistent(self) -> "Transaction":
        if self.txn_count_24h < self.txn_count_1h:
            raise ValueError("txn_count_24h must be greater than or equal to txn_count_1h")
        return self


class ScoreResponse(BaseModel):
    score: float
    threshold: float
    decision: Literal["flag", "allow"]
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transactions: list[Transaction] = Field(min_length=1, max_length=50)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
