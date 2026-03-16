from fastapi import APIRouter
from pydantic import BaseModel
import uuid
import datetime

# Triggering uvicorn hot-reload to catch newly installed dependencies
router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

#helper function to tell orchestrator which agent to run

def _detect_event_type(message: str) -> str:
    msg = message.lower()
    if any(k in msg for k in ["optimize", "best route", "reroute"]):
        return "optimize_request"
    if any(k in msg for k in ["sto", "classify", "transfer", "movement"]):
        return "sto_event"
    return "user_query"

class SessionSaveRequest(BaseModel):
    session_id: str
    title: str
    messages: list
    agent_id: str | None = None

SESSIONS_DB = {}

import sys
import os

# Inject the agents directory into the path so we can import the LangGraph orchestrator
agents_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../agents"))
if agents_dir not in sys.path:
    sys.path.append(agents_dir)

try:
    from orchestrator import Orchestrator
    orchestrator = Orchestrator()
except Exception as e:
    print(f"Warning: Orchestrator failed to load: {e}")
    orchestrator = None

@router.post("/")
async def chat(req: ChatRequest):
    if not orchestrator:
        return {"answer": "Error: Orchestrator offline.", "sources": []}
        
    query_lower = req.message.lower()
    
    # Simple NLP logic to determine if the user is asking about an STO/Classification or just chatting
    if any(keyword in query_lower for keyword in ["sto", "classify", "transfer", "order", "dc", "source", "destination"]):
        # It's an STO event, run through the original Orchestrator Graph
        dummy_sto = {
            "sto_id": f"MSG-{uuid.uuid4().hex[:6]}",
            "source_location": "DC_North" if "dc" in query_lower else "Unknown",
            "destination_location": "Store_44",
            "sku_id": "Laptops-X1" if "laptop" in query_lower else "Unknown",
            "quantity": 50
        }
        # detect user intent and route msg to correct agent
        # orchestrator has process_sto_event; route() existed as older API.
        final_state_state = orchestrator.process_sto_event(dummy_sto)
        sources = final_state_state.graph_context if hasattr(final_state_state, 'graph_context') else []
        if not sources:
            sources.append({
                "type": "neo4j",
                "source": "Agent Navigation",
                "confidence": 0.5,
                "text_snippet": "No distinct alternative graphs resolved."
            })
        answer = f"LangGraph STO Analysis complete. Classification: {final_state_state.classification}. Confidence: {final_state_state.confidence}. Reasoning: {final_state_state.reasoning_text}"
        return {"answer": answer, "sources": sources}
    else:
        # It's a general question/chat, use standard Conversational LLM or SQL Agent
        try:
            from langchain_openai import ChatOpenAI
            from langchain_community.utilities import SQLDatabase
            from langchain_community.agent_toolkits import create_sql_agent
            from langchain_core.messages import HumanMessage
            
            # Use the key from .env
            api_key = os.getenv("OPENAI_API_KEY")
            db_url = os.getenv("DATABASE_URL")
            if not api_key:
                 return {"answer": "I'm the SCNV Assistant. You need to configure my OPENAI_API_KEY in the backend `.env` file for conversational access, or ask me to explicitly classify an STO event by mentioning 'STO' or 'transfer'.", "sources": []}
            
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
            
            if db_url:
                db = SQLDatabase.from_uri(db_url)
                agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
                response = agent_executor.invoke({"input": f"You are the SCNV Assistant. A user says: '{req.message}'. Answer their question using the database if relevant. Be helpful and concise."})
                answer = response.get("output", "Sorry, I couldn't process that.")
                source_type = "sql_agent"
            else:
                prompt = f"You are the SCNV (Supply Chain Network Visibility) Assistant. A user says: '{req.message}'. Reply helpfully and concisely."
                response = llm([HumanMessage(content=prompt)])
                answer = response.content
                source_type = "llm"
            
            return {
                "answer": answer,
                "sources": [{
                    "type": source_type,
                    "source": "Generative Assistant with DB Access" if source_type == "sql_agent" else "Generative Assistant",
                    "confidence": 0.95,
                    "text_snippet": answer[:100] + "..." if len(answer) > 100 else answer
                }]
            }
        except Exception as e:
            return {"answer": f"Chat Engine failed: {str(e)}", "sources": []}

@router.get("/sessions")
async def get_sessions(agent_id: str | None = None):
    results = []
    for sid, data in SESSIONS_DB.items():
        if agent_id and data.get("agent_id") != agent_id:
            continue
        results.append({
            "id": str(sid),
            "timestamp": data["timestamp"],
            "title": data["title"],
            "agent_id": data.get("agent_id")
        })
    return {"sessions": results}

@router.post("/sessions/new")
async def save_session(req: SessionSaveRequest):
    SESSIONS_DB[req.session_id] = {
        "title": req.title,
        "messages": req.messages,
        "agent_id": req.agent_id,
        "timestamp": datetime.datetime.now().isoformat()
    }
    return {"status": "saved"}

@router.get("/sessions/{session_id}")
async def load_session(session_id: str):
    if session_id in SESSIONS_DB:
        return SESSIONS_DB[session_id]
    return {"messages": []}
