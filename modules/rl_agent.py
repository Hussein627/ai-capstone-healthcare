# ============================================================
# MODULE 10: Reinforcement Learning Agent
# Covers: Q-Learning for Treatment Optimization
# ============================================================

import numpy as np
from typing import Dict

class RLTreatmentOptimizer:
    """
    Q-Learning based Reinforcement Learning Agent.
    Recommends optimal care levels by mapping physical 
    states to actions using a pre-trained Q-Table.
    """
    def __init__(self):
        # The environment states based on patient vitals
        self.states = ['STABLE', 'GUARDED', 'SEVERE', 'CRITICAL']
        
        # The possible actions the agent can take
        self.actions = [
            'DISCHARGE_WITH_ADVICE', 
            'PRESCRIBE_MEDICATION', 
            'ADMIT_TO_WARD', 
            'RUSH_TO_ICU'
        ]
        
        # Simulated Q-Table (State x Action)
        # In a real environment, this starts as all zeros and is learned over 
        # thousands of episodes via the Bellman Equation. We are loading a 
        # "pre-trained" matrix to demonstrate inference.
        # Positive numbers are high rewards, negative are penalties (e.g., discharging a critical patient).
        self.q_table = {
            'STABLE':   [ 0.95,  0.20, -0.50, -1.00], 
            'GUARDED':  [ 0.10,  0.85,  0.30, -0.60], 
            'SEVERE':   [-0.80,  0.40,  0.90,  0.20], 
            'CRITICAL': [-1.00, -0.80,  0.50,  0.98]  
        }

    def _determine_state(self, temp: float, hr: int) -> str:
        """Map raw vitals to an RL environment state."""
        if hr >= 120 or temp >= 39.5:
            return 'CRITICAL'
        elif hr >= 100 or temp >= 38.5:
            return 'SEVERE'
        elif hr >= 90 or temp >= 37.8:
            return 'GUARDED'
        return 'STABLE'

    def get_optimal_action(self, state: str) -> str:
        """Policy: Choose the action with the highest Q-Value for the current state."""
        q_values = self.q_table.get(state, [0, 0, 0, 0])
        best_action_idx = np.argmax(q_values)
        return self.actions[best_action_idx]

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        
        # 1. Observe the environment state
        current_state = self._determine_state(percept.temperature, percept.heart_rate)
        
        # 2. Consult the Q-Table for the optimal policy
        best_action = self.get_optimal_action(current_state)
        
        # 3. Fetch the exact Q-Value confidence for reporting
        action_idx = self.actions.index(best_action)
        max_q_value = self.q_table[current_state][action_idx]
        
        return {
            'summary': f"RL Policy maps {current_state} state to {best_action} (Q-Value: {max_q_value})",
            'patient_state': current_state,
            'optimal_policy_action': best_action,
            'confidence': max_q_value
        }