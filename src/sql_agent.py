import os
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
import json

class QueryState(TypedDict):
    """State for the SQL query workflow."""
    question: str
    query: str
    result: str
    answer: str
    error: str

class SQLQueryAgent:
    def __init__(self, db_path: str = "db/products.db", model: str = "o4-mini"):
        """Initialize the SQL Query Agent."""
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatOpenAI(model=model, temperature=1)
        self.setup_tools()
        self.setup_workflow()
    
    def setup_tools(self):
        """Set up SQL database tools."""
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.tools = self.toolkit.get_tools()
        
        # Direct query execution tool
        self.query_tool = QuerySQLDatabaseTool(db=self.db)
    
    def setup_workflow(self):
        """Set up the LangGraph workflow."""
        # Create workflow
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("analyze_question", self.analyze_question)
        workflow.add_node("generate_query", self.generate_query)
        workflow.add_node("execute_query", self.execute_query)
        workflow.add_node("generate_answer", self.generate_answer)
        
        # Add edges
        workflow.add_edge(START, "analyze_question")
        workflow.add_edge("analyze_question", "generate_query")
        workflow.add_edge("generate_query", "execute_query")
        workflow.add_edge("execute_query", "generate_answer")
        
        # Compile workflow
        self.app = workflow.compile()
    
    def analyze_question(self, state: QueryState) -> QueryState:
        """Analyze the question to understand what data is needed."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert analyzing questions about a products database.
            The database has a 'products' table with columns:
            - id (INTEGER): Product ID
            - name (TEXT): Product name It is a unique identifier for each product. Product is nothing but a game playable in a casino. name can called as a product, game or name
            - description (TEXT): Product description
            - turnover (REAL): Product turnover/revenue
            - launch_date (DATE): Product launch date
            - country (TEXT): Country where product is available
            - segment (TEXT): Market segment (Low, Medium, High)
            
            Analyze the user's question and identify:
            1. What data they're looking for
            2. Any filters needed (country, date range, segment, etc.)
            3. Any aggregations needed (sum, average, count, group by)
            4. Any sorting or limiting needed
            """),
            ("human", "{question}")
        ])
        
        response = self.llm.invoke(prompt.format(question=state["question"]))
        return state
    
    def generate_query(self, state: QueryState) -> QueryState:
        """Generate SQL query from the question."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate a SQLite query for the products table.
            
            Table schema:
            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                turnover REAL,
                launch_date DATE,
                country TEXT,
                segment TEXT
            )
            
            Important notes:
            - Use DATE() function for date comparisons
            - Use datetime('now') for current date
            - For date ranges, use: date >= date('now', '-7 days')
            - Always use proper SQL syntax for SQLite
            - Include appropriate GROUP BY, ORDER BY, and LIMIT clauses
            - Return ONLY the SQL query, no explanations
            """),
            ("human", "Generate a SQL query for: {question}")
        ])
        
        response = self.llm.invoke(prompt.format(question=state["question"]))
        query = response.content.strip()
        
        # Clean up the query
        if query.startswith("```sql"):
            query = query[6:]
        if query.endswith("```"):
            query = query[:-3]
        query = query.strip()
        
        state["query"] = query
        return state
    
    def execute_query(self, state: QueryState) -> QueryState:
        """Execute the SQL query."""
        try:
            result = self.query_tool.invoke(state["query"])
            state["result"] = result
            state["error"] = ""
        except Exception as e:
            state["error"] = str(e)
            state["result"] = ""
            
            # Try to fix common errors
            if "no such column" in str(e).lower():
                # Retry with fixed column names
                fixed_query = self.fix_query(state["query"], str(e))
                try:
                    result = self.query_tool.invoke(fixed_query)
                    state["query"] = fixed_query
                    state["result"] = result
                    state["error"] = ""
                except Exception as e2:
                    state["error"] = f"Original error: {e}\nRetry error: {e2}"
        
        return state
    
    def fix_query(self, query: str, error: str) -> str:
        """Attempt to fix common SQL errors."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Fix the SQL query based on the error. 
            Common fixes:
            - Column names are case-sensitive
            - Date comparisons need proper formatting
            - Table name is 'products' (lowercase)
            
            Return ONLY the fixed SQL query."""),
            ("human", "Query: {query}\nError: {error}")
        ])
        
        response = self.llm.invoke(prompt.format(query=query, error=error))
        return response.content.strip()
    
    def generate_answer(self, state: QueryState) -> QueryState:
        """Generate natural language answer from query results."""
        if state.get("error"):
            state["answer"] = f"I encountered an error while processing your query: {state['error']}"
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that explains SQL query results in natural language.
            Given a question, SQL query, and results, provide a clear, concise answer.
            
            Guidelines:
            - Explain what the data shows
            - Highlight key findings
            - Use proper formatting for numbers and dates
            - If the result is empty, explain that no data matches the criteria
            """),
            ("human", """Question: {question}
            SQL Query: {query}
            Results: {result}
            
            Please provide a natural language answer. Don't make the response too long, be concise and to the point.""")
        ])
        
        response = self.llm.invoke(prompt.format(
            question=state["question"],
            query=state["query"],
            result=state["result"]
        ))
        
        state["answer"] = response.content
        return state
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Process a natural language query and return results."""
        initial_state = {
            "question": question,
            "query": "",
            "result": "",
            "answer": "",
            "error": ""
        }
        
        # Run the workflow
        final_state = self.app.invoke(initial_state)
        
        # Format response
        response = {
            "question": final_state["question"],
            "sql_query": final_state["query"],
            "raw_results": final_state["result"],
            "answer": final_state["answer"]
        }
        
        if final_state.get("error"):
            response["error"] = final_state["error"]
        
        return response

