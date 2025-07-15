import os
import uuid
from typing import List, TypedDict, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class RAGService:
    def __init__(self):
        # Initialize OpenAI embeddings and LLM
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
        
        # Initialize Chroma with persistent storage
        self.persist_directory = "./chroma_db"
        self.collection_name = "fdj-rag-analytics"
        
        # Create or load the vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        # Initialize data if needed
        self._ensure_data_loaded()
        
        # Create RAG app
        self.rag_app = self._create_rag_app()
    
    def _ensure_data_loaded(self):
        """Ensure data is loaded into vector store."""
        try:
            test_results = self.vector_store.similarity_search("test", k=1)
            if not test_results:
                print("RAG collection is empty. Ingesting data...")
                self._ingest_markdown_files('data/rag')
        except:
            print("RAG collection doesn't exist. Ingesting data...")
            self._ingest_markdown_files('data/rag')
    
    def _chunk_markdown(self, path):
        """Chunk markdown files by '---' separator."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = content.split('---\n')
        docs = []
        
        for chunk in chunks:
            lines = chunk.strip().split('\n')
            if not lines or not lines[0].startswith('# '):
                continue
            
            title = lines[0].replace('# ', '').strip()
            text = '\n'.join(lines[1:]).strip()
            
            doc = Document(
                page_content=text,
                metadata={
                    "title": title,
                    "source": path,
                    "chunk_id": f"{title.lower().replace(' ', '_')}_{uuid.uuid4()}"
                }
            )
            docs.append(doc)
        
        return docs
    
    def _ingest_markdown_files(self, directory):
        """Ingest all markdown files from directory."""
        all_documents = []
        
        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
            # Create sample rules.md file
            self._create_sample_rules(directory)
        
        for filename in os.listdir(directory):
            if filename.endswith('.md'):
                file_path = os.path.join(directory, filename)
                docs = self._chunk_markdown(file_path)
                all_documents.extend(docs)
                print(f"Processed {len(docs)} chunks from {filename}")
        
        if all_documents:
            ids = [doc.metadata['chunk_id'] for doc in all_documents]
            self.vector_store.add_documents(documents=all_documents, ids=ids)
            print(f"Total ingested: {len(all_documents)} documents")
    
    def _create_sample_rules(self, directory):
        """Create sample rules.md file."""
        sample_content = """# Lucky 7 Slots
- **Reels:** 3
- **Wild Symbol:** 7 expands to cover entire reel on a win
- **Max Payout:** 500× your stake
- **RTP:** 96.5%
- **Volatility:** Medium

---

# Roulette Pro
- **Wheel:** European (single-zero)
- **Side Bets:** Neighbours, Orphelins, Voisins du Zero
- **Min Bet:** $0.10
- **Max Bet:** $500
- **House Edge:** 2.7%

---

# Star Burst
- **Reels:** 5
- **Paylines:** 10 (both ways)
- **Wild Symbol:** Star (expands and triggers re-spin)
- **Max Win:** 50,000 coins
- **Features:** Win Both Ways, Expanding Wilds
"""
        
        with open(os.path.join(directory, "rules.md"), "w") as f:
            f.write(sample_content)
    
    def _create_rag_app(self):
        """Create the RAG application graph."""
        
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str
            persona: str
        
        def retrieve(state: State):
            """Retrieve relevant documents."""
            retrieved_docs = self.vector_store.similarity_search(
                state["question"],
                k=5
            )
            return {"context": retrieved_docs}
        
        def generate(state: State):
            """Generate answer with persona."""
            # Create persona-specific prompts
            persona_prompts = {
                "product_owner": """You are a technical product owner for a gaming platform. 
                Focus on system architecture, performance metrics, and technical implementation details.
                Discuss scalability, data structures, and engineering trade-offs when relevant.""",
                
                "marketing": """You are a marketing specialist for a gaming platform.
                Focus on user experience, engagement metrics, and player satisfaction.
                Use accessible language and highlight features that drive player retention and monetization."""
            }
            
            system_prompt = persona_prompts.get(state.get("persona", "product_owner"))
            
            # Build context
            enhanced_context = []
            for doc in state["context"]:
                title = doc.metadata.get("title", "Unknown")
                content = doc.page_content
                enhanced_context.append(f"Game: {title}\n{content}")
            
            full_context = "\n\n---\n\n".join(enhanced_context)
            
            prompt = PromptTemplate.from_template(
                system_prompt + """
                
                Use the following game information to answer the question.
                If the information doesn't contain the answer, say so clearly.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            messages = prompt.invoke({
                "context": full_context,
                "question": state["question"]
            })
            
            response = self.llm.invoke(messages.to_string())
            return {"answer": response.content}
        
        # Build graph
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", retrieve)
        graph_builder.add_node("generate", generate)
        
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        
        return graph_builder.compile()
    
    async def query_with_persona(self, question: str, persona: str = "product_owner") -> Dict[str, Any]:
        """Query the RAG system with persona."""
        result = await self.rag_app.ainvoke({
            "question": question,
            "persona": persona
        })
        
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": result.get("context", [])
        }
    
    def test_connection(self):
        """Test RAG service connection."""
        try:
            test_results = self.vector_store.similarity_search("test", k=1)
            return True
        except Exception as e:
            raise Exception(f"RAG service error: {str(e)}")