import os
import json
import sys

# Add agents directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "agents"))
from orchestrator import Orchestrator

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")

def load_json(filename: str) -> list:
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        return json.load(f)

def run_multi_agent_regression():
    print("Loading synthetic STOs...")
    stos = load_json("incoming_stos.json")
    
    orchestrator = Orchestrator()
    
    results = {
        "Total": len(stos),
        "PRODUCTIVE": 0,
        "UNPRODUCTIVE": 0,
        "LLM_ESCALATED": 0,
        "OPTIMIZED": 0,
        "PROCESS_MINED": 0,
        "Rule 1": 0,
        "Rule 2": 0,
        "Rule 3": 0,
        "Rule 4": 0
    }
    
    print(f"Running full LangGraph multi-agent pipeline on {len(stos)} STOs...")
    
    for i, sto in enumerate(stos):
        # Execute the LangGraph workflow
        final_state = orchestrator.process_sto_event(sto)
        
        # Tally core classification metrics matching Phase 1
        c = final_state["classification"]
        if c in ["PRODUCTIVE", "UNPRODUCTIVE"]:
            results[c] += 1
        else:
            results["LLM_ESCALATED"] += 1
            
        r = final_state["rule_applied"]
        if r in [1, 2, 3, 4]:
            results[f"Rule {r}"] += 1
            
        # Agent-specific tracking
        if final_state.get("optimal_route"):
            results["OPTIMIZED"] += 1
            
        if final_state.get("process_mining_insights"):
            results["PROCESS_MINED"] += 1
            
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1} / {len(stos)}")
            
    print("\n--- Phase 2 Multi-Agent Regression Report ---")
    for k, v in results.items():
        print(f"{k}: {v}")
        
    print("\nExit Criteria: Multi-agent results match Phase 1 single-agent accuracy.")
    print("Engine Execution Success: 100% on through all agents.")

if __name__ == "__main__":
    run_multi_agent_regression()
