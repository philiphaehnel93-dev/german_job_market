from pydantic import BaseModel


class SalaryMetrics(BaseModel):
    median: float | None = None
    p25: float | None = None  # 25. Percentile
    p75: float | None = None  # 75. Percentile
    raw_error: str | None = None