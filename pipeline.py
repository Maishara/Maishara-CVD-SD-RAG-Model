import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Import custom phase modules
from cvd_sd_model import BaselineCVDSystemDynamicsModel
from schemas import SystemDynamicsParameterSchema, RAGExtractionResponse
from rag_engine import ClinicalRAGManager
from ast_parser import ASTVerificationEngine, ASTVerificationError


class SystemDynamicsResearchPipeline:
    """
    Master research controller executing the full end-to-end pipeline:
    Prompt -> RAG Context -> Schema Validation -> AST Parsing -> PySD Execution -> Calibration Metrics
    """
    def __init__(self, vector_db_path: str = "./chroma_db"):
        self.rag_manager = ClinicalRAGManager(vector_db_path=vector_db_path)
        self.ast_engine = ASTVerificationEngine()
        
        # BHF Baseline Benchmarks for Validation (UK Cohort 40-70)
        self.uk_bhf_benchmarks = {
            "baseline_prevalence_pct": 6.8,  # 6.8% BHF benchmark
            "target_year_10_prevalence_pct": 6.2  # Expected target under 15% intervention
        }

    def generate_pysd_code_snippet(self, params: SystemDynamicsParameterSchema) -> str:
        """
        Dynamically constructs the executable Python code snippet 
        representing the PySD model instantiation and parameter assignments.
        """
        code_template = f"""
import numpy as np

# System Dynamics Rate Parameters (Extracted via RAG & Pydantic)
total_population = {params.total_population}
baseline_prevalence = {params.baseline_prevalence}
baseline_incidence_rate = {params.baseline_incidence_rate}
smoking_multiplier = {params.smoking_risk_multiplier}
hypertension_multiplier = {params.hypertension_risk_multiplier}
intervention_effect = {params.intervention_effect}

# Derived Initial Stocks & Effective Rate Assignments (Required by AST)
initial_cvd = total_population * baseline_prevalence
initial_susceptible = total_population - initial_cvd
effective_incidence_rate = baseline_incidence_rate * smoking_multiplier * hypertension_multiplier * (1.0 - intervention_effect)
dt = 0.1
time_horizon_years = {params.time_horizon_years}
"""
        return code_template.strip()

    def calculate_evaluation_metrics(self, simulation_results: pd.DataFrame) -> Dict[str, float]:
        """
        Computes formal empirical metrics (MAE & RMSE) comparing 
        model output against UK BHF benchmark prevalence targets.
        """
        simulated_initial = simulation_results['Prevalence_Percentage'].iloc[0]
        target_initial = self.uk_bhf_benchmarks['baseline_prevalence_pct']
        
        simulated_final = simulation_results['Prevalence_Percentage'].iloc[-1]
        target_final = self.uk_bhf_benchmarks['target_year_10_prevalence_pct']

        y_true = np.array([target_initial, target_final])
        y_pred = np.array([simulated_initial, simulated_final])

        # Calculate Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        pct_deviation = (abs(simulated_initial - target_initial) / target_initial) * 100.0

        return {
            "initial_prevalence_pct": round(simulated_initial, 3),
            "final_prevalence_pct": round(simulated_final, 3),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "baseline_pct_deviation": round(float(pct_deviation), 2)
        }

    def run_pipeline(self, user_scenario_prompt: str) -> Dict[str, Any]:
        """
        Executes the complete experimental workflow end-to-end.
        """
        print(f"\n==================================================")
        print(f"EXECUTING EXPERIMENTAL SCENARIO PIPELINE")
        print(f"Scenario: '{user_scenario_prompt}'")
        print(f"==================================================")

        # ----------------------------------------------------
        # STAGE 1: RAG Retrieval & Schema-Constrained Synthesis
        # ----------------------------------------------------
        print("\n[STAGE 1] Querying RAG Knowledge Store & Extracting Pydantic Schema...")
        rag_response: RAGExtractionResponse = self.rag_manager.extract_calibrated_parameters(user_scenario_prompt)
        params: SystemDynamicsParameterSchema = rag_response.parameters
        
        print("  ✓ Clinical Context Retrieved:")
        for chunk in rag_response.retrieved_context_chunks:
            print(f"    - {chunk}")
        print("  ✓ Schema Parameters Validated:")
        print(f"    - Baseline Prevalence: {params.baseline_prevalence * 100}%")
        print(f"    - Baseline Annual Incidence: {params.baseline_incidence_rate * 100}%")
        print(f"    - Smoking Risk Multiplier: {params.smoking_risk_multiplier}x")
        print(f"    - Intervention Effect: {params.intervention_effect * 100}% reduction")

        # ----------------------------------------------------
        # STAGE 2: Dynamic Code Generation & AST Verification
        # ----------------------------------------------------
        print("\n[STAGE 2] Generating PySD Differential Equation Code & Running AST Parser...")
        generated_code = self.generate_pysd_code_snippet(params)
        
        try:
            ast_report = self.ast_engine.verify_pysd_code(generated_code)
            print("  ✓ AST Verification PASSED:")
            print(f"    - Total AST Nodes Scanned: {ast_report['ast_nodes_count']}")
            print(f"    - Verified Required Variables: {ast_report['defined_variables']}")
        except ASTVerificationError as e:
            print(f"  ❌ AST Verification FAILED: {e}")
            return {"status": "FAILED_AST_VERIFICATION", "error": str(e)}

        # ----------------------------------------------------
        # STAGE 3: PySD Simulation Model Execution
        # ----------------------------------------------------
        print("\n[STAGE 3] Instantiating and Executing PySD Differential Model Engine...")
        model = BaselineCVDSystemDynamicsModel(
            total_population=params.total_population,
            baseline_prevalence=params.baseline_prevalence,
            baseline_incidence_rate=params.baseline_incidence_rate,
            smoking_multiplier=params.smoking_risk_multiplier,
            hypertension_multiplier=params.hypertension_risk_multiplier,
            intervention_effect=params.intervention_effect,
            time_horizon_years=params.time_horizon_years
        )
        
        results_df = model.run_simulation()
        print(f"  ✓ Simulation Execution Complete ({len(results_df)} timesteps computed).")

        # ----------------------------------------------------
        # STAGE 4: Empirical Calibration & Error Evaluation
        # ----------------------------------------------------
        print("\n[STAGE 4] Calculating Empirical Accuracy Metrics (MAE / RMSE)...")
        metrics = self.calculate_evaluation_metrics(results_df)
        
        print(f"  ✓ Empirical Results:")
        print(f"    - Year 0 Simulated Prevalence: {metrics['initial_prevalence_pct']}% (BHF Benchmark: 6.8%)")
        print(f"    - Year 10 Simulated Prevalence: {metrics['final_prevalence_pct']}%")
        print(f"    - Baseline % Deviation: {metrics['baseline_pct_deviation']}%")
        print(f"    - Mean Absolute Error (MAE): {metrics['mae']}")
        print(f"    - Root Mean Square Error (RMSE): {metrics['rmse']}")

        return {
            "status": "SUCCESS",
            "extracted_parameters": params.model_dump(),
            "simulation_results": results_df,
            "metrics": metrics,
            "ast_report": ast_report
        }


# Pipeline Execution Test
if __name__ == "__main__":
    # Ensure vector store has mock data initialized
    rag_init = ClinicalRAGManager()
    rag_init.populate_mock_knowledge_base()

    # Instantiate master controller
    pipeline = SystemDynamicsResearchPipeline()

    # Run scenario
    scenario_prompt = (
        "Simulate a 10-year intervention scenario where a smoking cessation program "
        "reduces CVD incidence in UK adults aged 40-70."
    )
    
    execution_report = pipeline.run_pipeline(scenario_prompt)
    
    if execution_report["status"] == "SUCCESS":
        print("\n==================================================")
        print("PIPELINE TEST PASSED SUCCESSFULLY!")
        print("==================================================")
        results_df = execution_report["simulation_results"]
        results_df.to_csv("pipeline_run_output.csv", index=False)
        print("\n✓ Results successfully saved to 'pipeline_run_output.csv'")