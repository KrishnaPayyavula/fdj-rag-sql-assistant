from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Literal, List
import os
from dotenv import load_dotenv
import asyncio

from src.database import DatabaseManager
from src.sql_agent import SQLQueryAgent
from src.rag_service import RAGService
from src.query_router import QueryRouter

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Hybrid RAG & Analytics Service")

# Initialize database if needed
db_manager = DatabaseManager()
if not os.path.exists("db/products.db"):
    print("Initializing database...")
    db_manager.create_schema()
    db_manager.load_data()

# Initialize services
sql_agent = SQLQueryAgent()
rag_service = RAGService()
query_router = QueryRouter()

class QueryRequest(BaseModel):
    question: str
    persona: Optional[Literal["product_owner", "marketing"]] = "product_owner"

class QueryResponse(BaseModel):
    question: str
    query_type: str
    answer: str
    sql_query: Optional[str] = None
    results: Optional[Any] = None
    context: Optional[List[Dict[str, str]]] = None
    error: Optional[str] = None

@app.get("/")
def read_root():
    return {
        "service": "Hybrid RAG & Analytics Service",
        "endpoints": {
            "/query": "POST - Process natural language queries with intelligent routing",
            "/health": "GET - Health check",
            "/examples": "GET - Example queries"
        }
    }

@app.get("/health")
def health_check():
    try:
        # Test database connection
        db_manager.test_connection()
        # Test RAG service
        rag_service.test_connection()
        return {"status": "healthy", "database": "connected", "rag": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query with intelligent routing."""
    try:
        # Determine query type
        query_type = await query_router.classify_query(request.question)
        
        response = {
            "question": request.question,
            "query_type": query_type,
        }
        
        if query_type == "analytics":
            # Use SQL agent for analytics queries
            sql_result = sql_agent.process_query(request.question)
            
            # Apply persona to the answer
            personalized_answer = await apply_persona(
                sql_result["answer"],
                request.persona,
                query_type,
                request.question
            )
            
            response.update({
                "answer": personalized_answer,
                "sql_query": sql_result["sql_query"],
                "results": parse_sql_results(sql_result.get("raw_results", ""))
            })
            
        elif query_type == "semantic":
            # Use RAG for semantic queries
            rag_result = await rag_service.query_with_persona(
                request.question,
                request.persona
            )
            
            response.update({
                "answer": rag_result["answer"],
                "context": [
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in rag_result.get("context", [])[:3]
                ]
            })
            
        else:  # general
            # Generate a general response
            general_answer = await generate_general_response(
                request.question,
                request.persona
            )
            response["answer"] = general_answer
        
        return QueryResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def apply_persona(answer: str, persona: str, query_type: str, question: str) -> str:
    """Apply persona styling to the answer."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    
    persona_prompts = {
        "product_owner": """You are a technical product owner. Rewrite this answer to:
        - Emphasize system architecture and technical implementation details
        - Discuss performance implications and trade-offs
        - Include metrics and data-driven insights
        - Use technical terminology appropriately
        - Focus on scalability and system design considerations""",
        
        "marketing": """You are a marketing specialist. Rewrite this answer to:
        - Highlight user experience and engagement metrics
        - Focus on conversion rates and customer value
        - Use persuasive and accessible language
        - Emphasize benefits and opportunities
        - Include actionable insights for marketing strategies"""
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona_prompts.get(persona, persona_prompts["product_owner"])),
        ("human", f"Original question: {question}\n\nOriginal answer: {answer}\n\nRewrite this answer according to your persona.")
    ])
    
    response = await llm.ainvoke(prompt.format())
    return response.content

async def generate_general_response(question: str, persona: str) -> str:
    """Generate a general response for non-specific queries."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
    
    persona_context = {
        "product_owner": "You are a technical product owner focused on system architecture and performance.",
        "marketing": "You are a marketing specialist focused on user engagement and conversion."
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{persona_context.get(persona, persona_context['product_owner'])} Answer the user's question helpfully and concisely."),
        ("human", "{question}")
    ])
    
    response = await llm.ainvoke(prompt.format(question=question))
    return response.content

def parse_sql_results(raw_results: str) -> Optional[List[Dict]]:
    """Parse SQL results into structured format."""
    if not raw_results:
        return None
    
    try:
        # Simple parsing for common result formats
        lines = raw_results.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Assume first line is headers
        headers = [h.strip() for h in lines[0].split('|')]
        results = []
        
        for line in lines[1:]:
            if line.strip():
                values = [v.strip() for v in line.split('|')]
                if len(values) == len(headers):
                    results.append(dict(zip(headers, values)))
        
        return results if results else None
    except:
        return None

@app.get("/examples")
def get_examples():
    """Return example queries to try."""
    return {
        "examples": {
            "analytics": [
                {
                    "question": "What is the total turnover by country?",
                    "persona": "product_owner",
                    "description": "SQL aggregation query with technical response"
                },
                {
                    "question": "Show me products with turnover greater than 1.0 in Belgium",
                    "persona": "marketing",
                    "description": "Filtered query with marketing-focused response"
                },
                {
                    "question": "What is the average turnover by segment over the past 7 days?",
                    "persona": "product_owner",
                    "description": "Time-based aggregation with technical insights"
                }
            ],
            "semantic": [
                {
                    "question": "How do I play Lucky 7 Slots?",
                    "persona": "marketing",
                    "description": "Game rules query with user-friendly explanation"
                },
                {
                    "question": "What are the payout rules for Roulette Pro?",
                    "persona": "product_owner",
                    "description": "Technical game mechanics query"
                },
                {
                    "question": "Explain the wild symbol mechanics in slot games",
                    "persona": "product_owner",
                    "description": "Cross-game feature explanation"
                }
            ],
            "general": [
                {
                    "question": "What makes a good casino game?",
                    "persona": "marketing",
                    "description": "General gaming industry question"
                },
                {
                    "question": "How do you measure game performance?",
                    "persona": "product_owner",
                    "description": "General analytics question"
                }
            ]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)