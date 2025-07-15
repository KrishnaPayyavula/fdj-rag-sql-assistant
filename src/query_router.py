from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class QueryClassification(BaseModel):
    query_type: Literal["analytics", "semantic", "general"]
    confidence: float
    reasoning: str

class QueryRouter:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=1)
        
    async def classify_query(self, question: str) -> str:
        """Classify query as analytics, semantic, or general."""
        
        structured_llm = self.llm.with_structured_output(QueryClassification)
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier for a gaming analytics platform.
            
            Classify queries into three categories:
            
            1. "analytics" - Queries about data, metrics, aggregations, or SQL-style questions
               Examples:
               - "What is the total turnover by country?"
               - "Show me products launched in 2023"
               - "Average revenue by segment"
               - "How many products are in Belgium?"
               - Questions with filters, grouping, counting, summing, etc.
            
            2. "semantic" - Queries about game rules, gameplay, features, or documentation
               Examples:
               - "How do I play Lucky 7 Slots?"
               - "What are the payout rules?"
               - "Explain the wild symbol"
               - "What side bets are available in Roulette?"
               - Questions about game mechanics or rules
            
            3. "general" - General questions not specific to data or game rules
               Examples:
               - "What makes a good game?"
               - "How do you design casino games?"
               - "What is the gaming industry like?"
               - Philosophical or open-ended questions
            
            Analyze the query and provide:
            - query_type: The classification
            - confidence: How confident you are (0.0 to 1.0)
            - reasoning: Brief explanation of your classification"""),
            ("human", "Classify this query: {question}")
        ])
        
        result = await structured_llm.ainvoke(
            classification_prompt.format(question=question)
        )
        
        # Log classification for debugging
        print(f"Query: {question}")
        print(f"Classification: {result.query_type} (confidence: {result.confidence})")
        print(f"Reasoning: {result.reasoning}")
        
        return result.query_type