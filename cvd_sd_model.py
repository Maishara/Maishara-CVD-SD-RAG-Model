import numpy as np
import pandas as pd

class BaselineCVDSystemDynamicsModel:
    """
    System Dynamics Model for CVD Population Trajectories.
    Models Susceptible (S) and CVD (C) stocks over a multi-year horizon.
    """
    def __init__(self, 
                 total_population: float = 100000.0,
                 baseline_prevalence: float = 0.068,  # 6.8% BHF baseline
                 baseline_incidence_rate: float = 0.008, # 0.8% annual new cases
                 smoking_multiplier: float = 1.5,
                 hypertension_multiplier: float = 1.4,
                 intervention_effect: float = 0.0, # 0.0 = no intervention, 0.15 = 15% reduction
                 natural_mortality_rate: float = 0.01,
                 cvd_mortality_rate: float = 0.035,
                 time_horizon_years: int = 10):
        
        # Simulation Parameters
        self.dt = 0.1  # Timestep in years
        self.steps = int(time_horizon_years / self.dt)
        self.time_horizon = time_horizon_years
        
        # Initial Stock Allocation
        self.initial_cvd = total_population * baseline_prevalence
        self.initial_susceptible = total_population - self.initial_cvd
        
        # Model Parameters
        self.baseline_incidence_rate = baseline_incidence_rate
        self.smoking_multiplier = smoking_multiplier
        self.hypertension_multiplier = hypertension_multiplier
        self.intervention_effect = intervention_effect
        self.natural_mortality_rate = natural_mortality_rate
        self.cvd_mortality_rate = cvd_mortality_rate

    def run_simulation(self) -> pd.DataFrame:
        """Executes differential rate equations over time."""
        # Tracking arrays
        time_points = np.linspace(0, self.time_horizon, self.steps)
        susceptible_stock = np.zeros(self.steps)
        cvd_stock = np.zeros(self.steps)
        
        # Set initial stock states
        susceptible_stock[0] = self.initial_susceptible
        cvd_stock[0] = self.initial_cvd
        
        # Calculate effective incidence rate parameter
        effective_incidence_rate = (
            self.baseline_incidence_rate * 
            self.smoking_multiplier * 
            self.hypertension_multiplier * 
            (1.0 - self.intervention_effect)
        )
        
        # Euler Integration Loop for Differential Equations
        for t in range(1, self.steps):
            S_prev = susceptible_stock[t-1]
            C_prev = cvd_stock[t-1]
            
            # Rate Equations (Flows)
            incidence_flow = S_prev * effective_incidence_rate
            susceptible_mortality = S_prev * self.natural_mortality_rate
            cvd_mortality = C_prev * (self.natural_mortality_rate + self.cvd_mortality_rate)
            
            # Stock Updates (dS/dt and dC/dt)
            dS = - incidence_flow - susceptible_mortality
            dC = incidence_flow - cvd_mortality
            
            # Euler Integration
            susceptible_stock[t] = max(0.0, S_prev + dS * self.dt)
            cvd_stock[t] = max(0.0, C_prev + dC * self.dt)
            
        # Format Results DataFrame
        results = pd.DataFrame({
            "Year": time_points,
            "Susceptible_Population": susceptible_stock,
            "CVD_Population": cvd_stock,
            "Total_Population": susceptible_stock + cvd_stock,
            "Prevalence_Percentage": (cvd_stock / (susceptible_stock + cvd_stock)) * 100
        })
        
        return results

# Test Run
if __name__ == "__main__":
    model = BaselineCVDSystemDynamicsModel(intervention_effect=0.15) # 15% intervention
    df = model.run_simulation()
    print("Simulation Complete!")
    print(df.head())
    print("\nFinal Prevalence in Year 10:", round(df['Prevalence_Percentage'].iloc[-1], 2), "%")