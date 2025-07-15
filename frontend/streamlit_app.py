import streamlit as st
import requests
import json
from typing import Dict, Any, Optional
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG & Analytics Service",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #0f172a;
    }
    
    /* Custom card styling */
    .custom-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
    }
    
    /* Query type badges */
    .query-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .analytics-badge {
        background-color: #6366f1;
        color: white;
    }
    
    .semantic-badge {
        background-color: #8b5cf6;
        color: white;
    }
    
    .general-badge {
        background-color: #10b981;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background-color: #334155;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background-color: #334155;
        border: 1px solid #475569;
        border-radius: 8px;
        color: #f1f5f9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e293b !important;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    
    /* Success/Error messages */
    .success-message {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #ef4444;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Context cards */
    .context-card {
        background-color: #334155;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .context-title {
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .spinner {
        border: 3px solid #334155;
        border-top: 3px solid #6366f1;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Markdown content styling */
    .markdown-content {
        color: #f1f5f9;
        line-height: 1.8;
    }
    
    .markdown-content strong {
        color: #f1f5f9;
        font-weight: 600;
    }
    
    .markdown-content ul, .markdown-content ol {
        margin: 1rem 0;
        padding-left: 2rem;
    }
    
    .markdown-content li {
        margin-bottom: 0.5rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'loading' not in st.session_state:
    st.session_state.loading = False

# API configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

def make_api_request(question: str, persona: str, model: str) -> Optional[Dict[str, Any]]:
    """Make API request to the backend service."""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "persona": persona,
                "model": model
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_loading_animation():
    """Display a custom loading animation."""
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="loading-spinner">
        <div class="spinner"></div>
    </div>
    <p style="text-align: center; color: #94a3b8; margin-top: 1rem;">
        Analyzing your query...
    </p>
    """, unsafe_allow_html=True)
    return loading_placeholder

def get_query_type_badge(query_type: str) -> str:
    """Get HTML for query type badge."""
    badges = {
        "analytics": '<div class="query-badge analytics-badge">📊 Analytics Query</div>',
        "semantic": '<div class="query-badge semantic-badge">📚 Semantic Query</div>',
        "general": '<div class="query-badge general-badge">💡 General Query</div>'
    }
    return badges.get(query_type, '<div class="query-badge general-badge">General Query</div>')

def display_response(response: Dict[str, Any]):
    """Display the API response in a structured format."""
    # Query type badge
    st.markdown(get_query_type_badge(response.get("query_type", "general")), unsafe_allow_html=True)
    
    # Original question
    st.markdown(f"**Question:** {response.get('question', '')}")
    
    # Main answer
    st.markdown("### Answer")
    st.markdown(f'<div class="markdown-content">{response.get("answer", "")}</div>', unsafe_allow_html=True)
    
    # SQL Query (if present)
    if response.get("sql_query"):
        st.markdown("### SQL Query")
        st.code(response["sql_query"], language="sql")
    
    # Context (if present)
    if response.get("context") and len(response["context"]) > 0:
        st.markdown("### Retrieved Context")
        for ctx in response["context"]:
            st.markdown(f"""
            <div class="context-card">
                <div class="context-title">{ctx.get('title', 'Unknown')}</div>
                <div>{ctx.get('content', '')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Results (if present)
    if response.get("results"):
        st.markdown("### Query Results")
        st.json(response["results"])

def main():
    # Header
    st.markdown("""
    <h1 style="text-align: center; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
    🎮 Hybrid RAG & Analytics Service
    </h1>
    <p style="text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;">
    Intelligent query routing for gaming analytics and documentation
    </p>
    """, unsafe_allow_html=True)
    
    # Main container
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Query Configuration")
        
        # Persona selection
        persona = st.selectbox(
            "Select Persona",
            options=["product_owner", "marketing"],
            format_func=lambda x: "Product Owner" if x == "product_owner" else "Marketing",
            help="Choose the perspective for the response"
        )
        
        # Model selection
        model = st.selectbox(
            "Select AI Model",
            options=["gpt-4o-mini"],
            format_func=lambda x: "GPT-4o Mini (Faster)",
            help="Choose the AI model for processing"
        )
        
        # Example queries
        st.markdown("### Example Queries")
        example_queries = {
            "Analytics": [
                "What is the total turnover by country?",
                "Show me products with turnover greater than 1.0",
                "What is the average turnover by segment?"
            ],
            "Semantic": [
                "How do I play Lucky 7 Slots?",
                "What are the payout rules for Roulette Pro?",
                "Explain the wild symbol mechanics"
            ],
            "General": [
                "What makes a good casino game?",
                "How do you measure game performance?"
            ]
        }
        
        for category, queries in example_queries.items():
            with st.expander(f"{category} Queries"):
                for query in queries:
                    if st.button(query, key=f"example_{query}"):
                        st.session_state.example_query = query
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Ask Your Question")
        
        # Query input
        default_query = st.session_state.get('example_query', '')
        question = st.text_area(
            "Enter your question",
            value=default_query,
            height=120,
            placeholder="Ask about game analytics, rules, or general gaming questions...",
            help="Type your question here"
        )
        
        # Submit button
        if st.button("Submit Query", type="primary", disabled=st.session_state.loading):
            if question.strip():
                st.session_state.loading = True
                loading_placeholder = display_loading_animation()
                
                # Make API request
                response = make_api_request(question, persona, model)
                
                # Clear loading animation
                loading_placeholder.empty()
                st.session_state.loading = False
                
                if response:
                    st.session_state.current_response = response
                    st.session_state.query_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": question,
                        "response": response
                    })
                    
                    # Clear example query
                    if 'example_query' in st.session_state:
                        del st.session_state.example_query
            else:
                st.warning("Please enter a question")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display response
        if st.session_state.current_response and not st.session_state.loading:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            display_response(st.session_state.current_response)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Query History (in sidebar)
    with st.sidebar:
        st.markdown("### Query History")
        if st.session_state.query_history:
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"{item['timestamp']} - {item['question'][:50]}..."):
                    if st.button(f"Load Query {i}", key=f"history_{i}"):
                        st.session_state.current_response = item['response']
                        st.rerun()
        else:
            st.info("No queries yet")
        
        # API Status
        st.markdown("### API Status")
        try:
            health_response = requests.get(f"{API_URL}/health", timeout=5)
            if health_response.status_code == 200:
                health_data = health_response.json()
                if health_data.get("status") == "healthy":
                    st.success("✅ API Connected")
                else:
                    st.error("❌ API Unhealthy")
            else:
                st.error("❌ API Unreachable")
        except:
            st.error("❌ API Offline")

if __name__ == "__main__":
    main()