from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class AnalysisPlan(BaseModel):
    analysis_goal: str
    priority_dimensions: list[str]
    relevant_analyses: list[Literal["sales", "product", "customer"]]
    requested_period: str | None


class BusinessInsights(BaseModel):
    key_finding: str
    supporting_evidence: list[str]
    possible_drivers: list[str]
    important_caveats: list[str]


class Recommendation(BaseModel):
    action: str
    evidence: list[str]
    rationale: str
    validation_needed: str | None = None

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class Recommendations(BaseModel):
    recommendations: list[Recommendation]
    overall_caveat: str | None = None
