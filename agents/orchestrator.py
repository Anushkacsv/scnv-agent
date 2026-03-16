import os
import sys
from typing import TypedDict, Annotated, Sequence
import operator

# LangGraph specifics
from langgraph.graph import StateGraph, END

# Import the protocol
sys.path.append(os.path.dirname(__file__))
from protocol import AgentState, STOEvent

# Import Agents
from scm_analyst import SCMAnalystAgent
from process_mining import ProcessMiningAgent
from optimizer import OptimizerAgent
from llm_engine import LLMEngine

from neo4j_nodes import Neo4jMemoryNode

class Orchestrator:
    """
    The central LangGraph orchestrator that routes STO events through the multi-agent system.
    """
    def __init__(self):
        self.scm_analyst = SCMAnalystAgent()
        self.process_mining = ProcessMiningAgent()
        self.optimizer = OptimizerAgent()
        self.llm_engine = LLMEngine()
        self.neo4j_node = Neo4jMemoryNode()
        
        # Build the graph
        self.workflow = StateGraph(AgentState)
        
        # Add Nodes (avoid naming collisions with state keys)
        self.workflow.add_node("neo4j_context", self.neo4j_node.retrieve_graph_context)
        self.workflow.add_node("scm_analyst", self.scm_analyst.invoke)
        self.workflow.add_node("process_mining", self.process_mining.invoke)
        self.workflow.add_node("optimizer", self.optimizer.invoke)
        self.workflow.add_node("tier2_llm", self.tier2_escalation)
        
        # Define Edges / Routing
        # Start by querying Neo4j for network relationships
        self.workflow.set_entry_point("neo4j_context")
        self.workflow.add_edge("neo4j_context", "scm_analyst")
        
        # From SCM Analyst, determine if we need Tier 2 LLM or if we can proceed to Optimizer/Mining
        self.workflow.add_conditional_edges(
            "scm_analyst",
            self.route_after_classification,
            {
                "optimizer": "optimizer",
                "tier2_llm": "tier2_llm"
            }
        )
        
        # Tier 2 LLM goes to Optimizer
        self.workflow.add_edge("tier2_llm", "optimizer")
        
        # Optimizer goes to Process Mining (which is toggleable inside)
        self.workflow.add_edge("optimizer", "process_mining")
        
        # Process Mining ends the flow
        self.workflow.add_edge("process_mining", END)
        
        # Compile graph
        self.app = self.workflow.compile()
        
    def tier2_escalation(self, state: AgentState) -> AgentState:
        """
        Node that triggers the LLM Engine if confidence is too low.
        """
        res = self.llm_engine.generate_classification(state.sto, "Tier 1 rules failed.")
        state.classification = res["classification"]
        state.rule_applied = res["rule_applied"]
        state.root_cause = res["root_cause"]
        state.confidence = res["confidence"]
        state.reasoning_text = res["reasoning_text"]
        return state

    def route_after_classification(self, state: AgentState) -> str:
        """
        Routing logic post-SCM Analyst.
        """
        if state.classification == "UNKNOWN" or state.confidence < 0.8:
            return "tier2_llm"
        else:
            return "optimizer"
            
    def process_sto_event(self, sto_dict: dict) -> AgentState:
        """
        Entry point to execute the graph for a single STO.
        """
        initial_state = AgentState(sto=sto_dict)
        final_state = self.app.invoke(initial_state)
        return final_state
