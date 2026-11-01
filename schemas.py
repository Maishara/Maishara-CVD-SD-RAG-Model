from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class SystemDynamicsParameterSchema(BaseModel):
    """
    Pydantic schema enforcing structural and numeric bounds on
    LLM-generated System Dynamics parameters.
    """
    target_disease: str = Field(
        ..., description="Name of the targeted condition (e.g., Cardiovascular Disease)"
    )
    total_population: float = Field(
        100000.0, gt=0, description="Total baseline cohort population size"
    )
    
    # Grounded Baseline Parameters (Extracted via RAG)
    baseline_prevalence: float = Field(
        ..., ge=0.001, le=0.50, 
        description="Initial disease prevalence proportion (e.g., 0.068 for 6.8% BHF baseline)"
    )
    baseline_incidence_rate: float = Field(
        ..., ge=0.0001, le=0.10, 
        description="Annual incidence rate of new cases in susceptible population"
    )
    
    # Lifestyle Multipliers (Extracted via RAG)
    smoking_risk_multiplier: float = Field(
        1.0, ge=1.0, le=5.0, 
        description="Relative risk multiplier for smoking"
    )
    hypertension_risk_multiplier: float = Field(
        1.0, ge=1.0, le=5.0, 
        description="Relative risk multiplier for high blood pressure"
    )
    
    # Intervention Scenario Parameters
    intervention_effect: float = Field(
        0.0, ge=0.0, le=0.90, 
        description="Proportion reduction in incidence from intervention (0.0 to 0.90)"
    )
    time_horizon_years: int = Field(
        10, ge=1, le=50, 
        description="Simulation duration in years"
    )
    
    # Data Provenance Tracking
    data_sources_cited: List[str] = Field(
        default_factory=list, 
        description="Citations extracted from RAG context (e.g., BHF Compendium 2025, NHS QOF)"
    )

    @field_validator('baseline_prevalence')
    @classmethod
    def validate_prevalence_bounds(cls, v: float) -> float:
        if v > 0.30:
            raise ValueError("Prevalence exceeding 30% is epidemiologically implausible for general CVD cohort.")
        return v

class RAGExtractionResponse(BaseModel):
    """Container for RAG retrieval results and populated parameter schema."""
    retrieved_context_chunks: List[str]
    parameters: SystemDynamicsParameterSchema