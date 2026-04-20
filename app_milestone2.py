import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, TypedDict
import json
from datetime import datetime

# LangGraph and LangChain imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings
    import chromadb
    from chromadb.config import Settings
    from pydantic import BaseModel
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Agentic Assessment Design - Milestone 2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #e74c3c;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .step-card {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Assessment State for LangGraph
class AssessmentState(TypedDict):
    question_text: str
    subject: str
    cognitive_level: str
    difficulty_prediction: Optional[str]
    confidence_scores: Optional[Dict[str, float]]
    analysis: Optional[str]
    improvements: Optional[List[str]]
    pedagogical_context: Optional[str]
    final_recommendations: Optional[str]
    step_history: Optional[List[str]]

class AgenticAssessmentSystem:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.llm = None
        self.embeddings = None
        self.chroma_client = None
        self.knowledge_base = None
        self.graph = None
        
    def initialize_ml_models(self):
        """Load ML models from Milestone 1"""
        try:
            self.model = joblib.load('models/difficulty_model.pkl')
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            return True
        except FileNotFoundError:
            return False
    
    def initialize_llm(self, provider="ollama", model_name="mistral"):
        """Initialize LLM based on provider"""
        try:
            if provider == "ollama":
                self.llm = Ollama(model=model_name, base_url="http://localhost:11434")
                self.embeddings = OllamaEmbeddings(model=model_name)
            elif provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return False
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
                self.embeddings = OpenAIEmbeddings(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return False
    
    def initialize_knowledge_base(self):
        """Initialize Chroma vector database with pedagogical knowledge"""
        try:
            # Create Chroma client
            self.chroma_client = chromadb.Client(Settings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            try:
                self.knowledge_base = self.chroma_client.get_collection("pedagogical_knowledge")
            except:
                self.knowledge_base = self.chroma_client.create_collection("pedagogical_knowledge")
                self._populate_knowledge_base()
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize knowledge base: {str(e)}")
            return False
    
    def _populate_knowledge_base(self):
        """Populate knowledge base with pedagogical documents"""
        pedagogical_docs = [
            {
                "id": "blooms_taxonomy",
                "content": """Bloom's Taxonomy provides a framework for categorizing educational goals and learning objectives:
                
                1. Remember: Recall facts and basic concepts
                2. Understand: Explain ideas or concepts  
                3. Apply: Use information in new situations
                4. Analyze: Draw connections among ideas
                5. Evaluate: Justify a stand or decision
                6. Create: Produce new or original work
                
                Questions should align with appropriate cognitive levels for the target audience.""",
                "metadata": {"category": "cognitive_levels", "importance": "high"}
            },
            {
                "id": "question_design",
                "content": """Best practices for question design:
                
                1. Clarity: Questions should be unambiguous and clearly worded
                2. Alignment: Questions should match learning objectives
                3. Appropriate difficulty: Match student level and course goals
                4. Discrimination: Good questions differentiate between high and low performers
                5. Avoid bias: Questions should be fair and accessible to all students
                6. Single focus: Each question should test one concept clearly""",
                "metadata": {"category": "design_principles", "importance": "high"}
            },
            {
                "id": "difficulty_calibration",
                "content": """Difficulty calibration guidelines:
                
                Easy questions (70-90% success rate):
                - Basic recall and recognition
                - Fundamental concepts
                - Direct application of formulas
                
                Medium questions (40-70% success rate):
                - Application of concepts
                - Multi-step problems
                - Analysis and synthesis
                
                Hard questions (10-40% success rate):
                - Complex problem solving
                - Creative application
                - Evaluation and critique""",
                "metadata": {"category": "difficulty", "importance": "high"}
            },
            {
                "id": "learning_gaps",
                "content": """Identifying and addressing learning gaps:
                
                1. Common misconceptions: Identify frequent student errors
                2. Prerequisite knowledge: Ensure foundational concepts are solid
                3. Scaffolding: Provide intermediate steps for complex problems
                4. Multiple representations: Use various ways to present concepts
                5. Feedback: Provide specific, actionable feedback
                6. Remediation: Offer additional practice for struggling areas""",
                "metadata": {"category": "learning_support", "importance": "medium"}
            },
            {
                "id": "assessment_quality",
                "content": """Assessment quality metrics:
                
                1. Validity: Does the assessment measure what it claims to measure?
                2. Reliability: Are results consistent across time and conditions?
                3. Discrimination index: Do questions differentiate between students?
                4. Item difficulty: Is the difficulty appropriate for the population?
                5. Distractor analysis: Are wrong answers plausible and informative?
                6. Bias review: Are questions fair to all demographic groups?""",
                "metadata": {"category": "quality_metrics", "importance": "high"}
            }
        ]
        
        # Add documents to knowledge base
        for doc in pedagogical_docs:
            self.knowledge_base.add(
                documents=[doc["content"]],
                metadatas=[doc["metadata"]],
                ids=[doc["id"]]
            )
    
    def predict_difficulty(self, question_text):
        """Predict difficulty using ML model"""
        if not self.model or not self.vectorizer:
            return None
            
        features = self.vectorizer.transform([question_text])
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        difficulty_map = {0: "Easy", 1: "Medium", 2: "Hard"}
        
        return {
            'difficulty': difficulty_map[prediction],
            'confidence': max(probabilities),
            'probabilities': {
                'Easy': probabilities[0],
                'Medium': probabilities[1],
                'Hard': probabilities[2]
            }
        }
    
    def analyze_node(self, state: AssessmentState) -> AssessmentState:
        """Analysis node: Predict difficulty and extract metrics"""
        question_text = state["question_text"]
        
        # ML prediction
        ml_result = self.predict_difficulty(question_text)
        
        # Text analysis
        words = question_text.split()
        analysis = f"""
        Question Analysis:
        - Word count: {len(words)}
        - Character count: {len(question_text)}
        - Predicted difficulty: {ml_result['difficulty'] if ml_result else 'Unknown'}
        - Confidence: {ml_result['confidence']:.2f if ml_result else 0}
        - Subject: {state.get('subject', 'Not specified')}
        - Cognitive level: {state.get('cognitive_level', 'Not specified')}
        """
        
        state["difficulty_prediction"] = ml_result['difficulty'] if ml_result else "Unknown"
        state["confidence_scores"] = ml_result['probabilities'] if ml_result else {}
        state["analysis"] = analysis
        state["step_history"] = state.get("step_history", []) + ["Analysis completed"]
        
        return state
    
    def retrieve_node(self, state: AssessmentState) -> AssessmentState:
        """Retrieval node: Query pedagogical knowledge base"""
        if not self.knowledge_base:
            state["pedagogical_context"] = "Knowledge base not available"
            return state
        
        # Query based on difficulty and subject
        query = f"question design {state.get('difficulty_prediction', '')} {state.get('subject', '')}"
        
        try:
            results = self.knowledge_base.query(
                query_texts=[query],
                n_results=3
            )
            
            context = "Relevant pedagogical guidance:\n\n"
            for i, doc in enumerate(results['documents'][0]):
                context += f"{i+1}. {doc[:200]}...\n\n"
            
            state["pedagogical_context"] = context
        except Exception as e:
            state["pedagogical_context"] = f"Error retrieving context: {str(e)}"
        
        state["step_history"] = state.get("step_history", []) + ["Knowledge retrieval completed"]
        return state
    
    def reason_node(self, state: AssessmentState) -> AssessmentState:
        """Reasoning node: Generate improvement suggestions"""
        if not self.llm:
            state["improvements"] = ["LLM not available for reasoning"]
            return state
        
        prompt = f"""
        As an educational assessment expert, analyze this exam question and provide improvement suggestions:
        
        Question: {state['question_text']}
        Subject: {state.get('subject', 'Not specified')}
        Predicted Difficulty: {state.get('difficulty_prediction', 'Unknown')}
        Confidence: {state.get('confidence_scores', {}).get(state.get('difficulty_prediction', ''), 0):.2f}
        
        Pedagogical Context:
        {state.get('pedagogical_context', 'No context available')}
        
        Please provide 3-5 specific, actionable improvement suggestions for this question.
        Focus on clarity, alignment with learning objectives, and appropriate difficulty level.
        
        Format your response as a numbered list.
        """
        
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                improvements_text = response.content
            else:
                improvements_text = str(response)
            
            # Parse improvements into list
            improvements = []
            for line in improvements_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    improvements.append(line)
            
            state["improvements"] = improvements if improvements else [improvements_text]
        except Exception as e:
            state["improvements"] = [f"Error generating improvements: {str(e)}"]
        
        state["step_history"] = state.get("step_history", []) + ["Reasoning completed"]
        return state
    
    def validate_node(self, state: AssessmentState) -> AssessmentState:
        """Validation node: Check quality and generate final recommendations"""
        
        # Generate final recommendations
        recommendations = f"""
        ## Assessment Quality Report
        
        **Question**: {state['question_text'][:100]}...
        
        **Predicted Difficulty**: {state.get('difficulty_prediction', 'Unknown')}
        **Confidence Score**: {state.get('confidence_scores', {}).get(state.get('difficulty_prediction', ''), 0):.1%}
        
        **Analysis Summary**:
        {state.get('analysis', 'No analysis available')}
        
        **Improvement Suggestions**:
        """
        
        for i, improvement in enumerate(state.get('improvements', []), 1):
            recommendations += f"\n{i}. {improvement}"
        
        recommendations += f"""
        
        **Pedagogical Alignment**:
        Based on educational best practices, this question shows potential for improvement in clarity and alignment with learning objectives.
        
        **Next Steps**:
        1. Review and implement suggested improvements
        2. Pilot test with a small group of students
        3. Analyze student responses for further refinement
        4. Consider alignment with course learning objectives
        
        **Disclaimer**: This analysis is AI-generated and should be reviewed by educational experts before implementation.
        """
        
        state["final_recommendations"] = recommendations
        state["step_history"] = state.get("step_history", []) + ["Validation completed"]
        
        return state
    
    def build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AssessmentState)
        
        # Add nodes
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("reason", self.reason_node)
        workflow.add_node("validate", self.validate_node)
        
        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "reason")
        workflow.add_edge("reason", "validate")
        workflow.add_edge("validate", END)
        
        self.graph = workflow.compile()
    
    def run_assessment(self, question_text, subject="", cognitive_level=""):
        """Run the complete agentic assessment workflow"""
        if not self.graph:
            return None
        
        initial_state = AssessmentState(
            question_text=question_text,
            subject=subject,
            cognitive_level=cognitive_level,
            difficulty_prediction=None,
            confidence_scores=None,
            analysis=None,
            improvements=None,
            pedagogical_context=None,
            final_recommendations=None,
            step_history=[]
        )
        
        try:
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            st.error(f"Error running assessment: {str(e)}")
            return None

@st.cache_resource
def initialize_system():
    """Initialize the agentic assessment system"""
    system = AgenticAssessmentSystem()
    
    # Initialize ML models
    if not system.initialize_ml_models():
        st.warning("ML models not found. Please run `python train_model.py` first.")
    
    # Initialize knowledge base
    if not system.initialize_knowledge_base():
        st.warning("Could not initialize knowledge base.")
    
    return system

def main():
    # Header
    st.markdown('<h1 class="main-header">Agentic Assessment Design Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Milestone 2: AI-Powered Assessment Improvement with LangGraph</p>', unsafe_allow_html=True)
    
    # Check if LangGraph is available
    if not LANGGRAPH_AVAILABLE:
        st.error("LangGraph dependencies not installed. Please run: `pip install langgraph langchain chromadb`")
        return
    
    # Initialize system
    system = initialize_system()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # LLM Provider selection
    llm_provider = st.sidebar.selectbox(
        "LLM Provider:",
        ["ollama", "openai", "none"],
        help="Choose your LLM provider. Ollama requires local installation."
    )
    
    if llm_provider == "ollama":
        model_name = st.sidebar.selectbox(
            "Ollama Model:",
            ["mistral", "llama2", "neural-chat"],
            help="Select the Ollama model to use"
        )
        if st.sidebar.button("Initialize Ollama"):
            if system.initialize_llm("ollama", model_name):
                st.sidebar.success("Ollama initialized")
            else:
                st.sidebar.error("Failed to initialize Ollama")
    
    elif llm_provider == "openai":
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            if st.sidebar.button("Initialize OpenAI"):
                if system.initialize_llm("openai"):
                    st.sidebar.success("OpenAI initialized")
                else:
                    st.sidebar.error("Failed to initialize OpenAI")
    
    # Build graph
    if st.sidebar.button("Build Agent Workflow"):
        system.build_graph()
        st.sidebar.success("Agent workflow built")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["Agentic Analysis", "Workflow Visualization", "Knowledge Base"])
    
    with tab1:
        agentic_analysis_interface(system)
    
    with tab2:
        workflow_visualization()
    
    with tab3:
        knowledge_base_interface(system)

def agentic_analysis_interface(system):
    """Main agentic analysis interface"""
    st.header("Autonomous Assessment Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question_text = st.text_area(
            "Enter your exam question:",
            height=150,
            placeholder="Example: Analyze the impact of climate change on global food security and propose three evidence-based solutions."
        )
        
        col_meta1, col_meta2 = st.columns(2)
        with col_meta1:
            subject = st.selectbox(
                "Subject:",
                ["", "Mathematics", "Physics", "Computer Science", "Engineering", "Biology", "Chemistry", "Environmental Science"]
            )
        with col_meta2:
            cognitive_level = st.selectbox(
                "Target Cognitive Level:",
                ["", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            )
    
    with col2:
        st.markdown("### Agent Status")
        
        # System status
        ml_status = "✓" if system.model and system.vectorizer else "✗"
        llm_status = "✓" if system.llm else "✗"
        kb_status = "✓" if system.knowledge_base else "✗"
        graph_status = "✓" if system.graph else "✗"
        
        st.markdown(f"""
        **ML Models**: {ml_status}  
        **LLM**: {llm_status}  
        **Knowledge Base**: {kb_status}  
        **Agent Graph**: {graph_status}
        """)
    
    # Analysis button
    if st.button("Run Agentic Analysis", type="primary", use_container_width=True):
        if not question_text.strip():
            st.warning("Please enter a question to analyze.")
            return
        
        if not system.graph:
            st.error("Agent workflow not built. Please build the workflow first.")
            return
        
        with st.spinner("Agent is analyzing your question..."):
            result = system.run_assessment(question_text, subject, cognitive_level)
            
            if result:
                display_agentic_results(result)

def display_agentic_results(result):
    """Display the results from agentic analysis"""
    st.markdown("---")
    st.header("Agentic Analysis Results")
    
    # Workflow steps
    st.subheader("Agent Workflow Steps")
    steps = result.get("step_history", [])
    for i, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-card">
            <strong>Step {i}:</strong> {step}
        </div>
        """, unsafe_allow_html=True)
    
    # Main results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ML Analysis")
        difficulty = result.get("difficulty_prediction", "Unknown")
        confidence_scores = result.get("confidence_scores", {})
        
        st.markdown(f"""
        <div class="agent-card">
            <h3>Predicted Difficulty: {difficulty}</h3>
            <p>Confidence Scores:</p>
            <ul>
                <li>Easy: {confidence_scores.get('Easy', 0):.1%}</li>
                <li>Medium: {confidence_scores.get('Medium', 0):.1%}</li>
                <li>Hard: {confidence_scores.get('Hard', 0):.1%}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Pedagogical Context")
        context = result.get("pedagogical_context", "No context available")
        st.text_area("Retrieved Knowledge:", context[:500] + "...", height=200, disabled=True)
    
    # Improvements
    st.subheader("AI-Generated Improvements")
    improvements = result.get("improvements", [])
    for i, improvement in enumerate(improvements, 1):
        st.markdown(f"""
        <div class="recommendation-box">
            <strong>Suggestion {i}:</strong> {improvement}
        </div>
        """, unsafe_allow_html=True)
    
    # Final recommendations
    st.subheader("Complete Assessment Report")
    recommendations = result.get("final_recommendations", "No recommendations available")
    st.markdown(recommendations)
    
    # Download report
    report_data = {
        "question": result.get("question_text", ""),
        "difficulty": result.get("difficulty_prediction", ""),
        "confidence_scores": result.get("confidence_scores", {}),
        "improvements": result.get("improvements", []),
        "timestamp": datetime.now().isoformat()
    }
    
    st.download_button(
        label="Download Assessment Report",
        data=json.dumps(report_data, indent=2),
        file_name=f"assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def workflow_visualization():
    """Visualize the agentic workflow"""
    st.header("Agent Workflow Visualization")
    
    st.markdown("""
    ## LangGraph Workflow Architecture
    
    The agentic assessment system follows a multi-step reasoning process:
    """)
    
    # Workflow diagram
    st.markdown("""
    ```
    Question Input
           ↓
    [ANALYZE NODE]
       • ML difficulty prediction
       • Text metrics extraction
       • Initial assessment
           ↓
    [RETRIEVE NODE]
       • Query knowledge base
       • Semantic search
       • Pedagogical context
           ↓
    [REASON NODE]
       • LLM-powered analysis
       • Improvement generation
       • Prioritization
           ↓
    [VALIDATE NODE]
       • Quality check
       • Final recommendations
       • Report generation
           ↓
    Assessment Report
    ```
    """)
    
    # Node descriptions
    st.subheader("Node Descriptions")
    
    nodes = [
        {
            "name": "Analyze Node",
            "description": "Uses trained ML models to predict question difficulty and extract text metrics",
            "inputs": "Question text, subject, cognitive level",
            "outputs": "Difficulty prediction, confidence scores, text analysis"
        },
        {
            "name": "Retrieve Node", 
            "description": "Queries the pedagogical knowledge base using semantic search",
            "inputs": "Question context, difficulty prediction",
            "outputs": "Relevant pedagogical guidance and best practices"
        },
        {
            "name": "Reason Node",
            "description": "Uses LLM to generate specific improvement suggestions",
            "inputs": "Question analysis, pedagogical context",
            "outputs": "Prioritized improvement suggestions"
        },
        {
            "name": "Validate Node",
            "description": "Validates suggestions and generates final assessment report",
            "inputs": "All previous analysis and suggestions",
            "outputs": "Complete assessment report with recommendations"
        }
    ]
    
    for node in nodes:
        st.markdown(f"""
        <div class="step-card">
            <h4>{node['name']}</h4>
            <p><strong>Description:</strong> {node['description']}</p>
            <p><strong>Inputs:</strong> {node['inputs']}</p>
            <p><strong>Outputs:</strong> {node['outputs']}</p>
        </div>
        """, unsafe_allow_html=True)

def knowledge_base_interface(system):
    """Interface for exploring the knowledge base"""
    st.header("Pedagogical Knowledge Base")
    
    st.markdown("""
    The knowledge base contains curated pedagogical content to guide assessment improvement:
    """)
    
    # Knowledge categories
    categories = [
        {
            "name": "Bloom's Taxonomy",
            "description": "Cognitive level framework for learning objectives",
            "content": "Guidelines for aligning questions with appropriate cognitive levels"
        },
        {
            "name": "Question Design Principles",
            "description": "Best practices for creating effective assessment questions",
            "content": "Clarity, alignment, appropriate difficulty, discrimination, bias avoidance"
        },
        {
            "name": "Difficulty Calibration",
            "description": "Guidelines for setting appropriate question difficulty",
            "content": "Target success rates and complexity guidelines for different levels"
        },
        {
            "name": "Learning Gap Analysis",
            "description": "Strategies for identifying and addressing student learning gaps",
            "content": "Common misconceptions, prerequisite knowledge, scaffolding techniques"
        },
        {
            "name": "Assessment Quality Metrics",
            "description": "Criteria for evaluating assessment effectiveness",
            "content": "Validity, reliability, discrimination index, item difficulty analysis"
        }
    ]
    
    for category in categories:
        with st.expander(f"{category['name']}"):
            st.markdown(f"**Description:** {category['description']}")
            st.markdown(f"**Content:** {category['content']}")
    
    # Query interface
    st.subheader("Query Knowledge Base")
    query = st.text_input("Enter your pedagogical question:")
    
    if query and system.knowledge_base:
        try:
            results = system.knowledge_base.query(
                query_texts=[query],
                n_results=3
            )
            
            st.subheader("Search Results")
            for i, doc in enumerate(results['documents'][0], 1):
                st.markdown(f"""
                <div class="recommendation-box">
                    <h4>Result {i}</h4>
                    <p>{doc}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error querying knowledge base: {str(e)}")
    
    # Responsible AI disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <h4>Responsible AI Notice</h4>
        <p>This AI system provides suggestions to assist educators in assessment design. 
        All recommendations should be reviewed by qualified educational professionals before implementation. 
        The system may have limitations and biases that require human oversight.</p>
        
        <p><strong>Key Principles:</strong></p>
        <ul>
            <li>AI assists but does not replace educator judgment</li>
            <li>Recommendations require professional validation</li>
            <li>System limitations should be acknowledged</li>
            <li>Student privacy and fairness must be maintained</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()