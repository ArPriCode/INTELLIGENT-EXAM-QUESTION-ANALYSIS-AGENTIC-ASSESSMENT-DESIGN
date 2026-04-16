# Milestone 2: Agentic AI Assessment Design Assistant

## Overview

Milestone 2 extends the ML-based exam question analytics system into an autonomous agentic AI application using **LangGraph** and **RAG** (Retrieval-Augmented Generation). The system now reasons about assessment quality, retrieves pedagogical best practices, and generates structured improvement recommendations.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)               │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼────┐            ┌──────▼──────┐
    │ ML     │            │ LangGraph   │
    │ Models │            │ Agent       │
    └───┬────┘            └──────┬──────┘
        │                        │
        │    ┌───────────────────┤
        │    │                   │
    ┌───▼────▼──┐        ┌──────▼──────┐
    │ TF-IDF    │        │ RAG System  │
    │ Vectorizer│        │ (Chroma)    │
    └───────────┘        └──────┬──────┘
                                │
                        ┌───────▼────────┐
                        │ Pedagogical    │
                        │ Knowledge Base │
                        └────────────────┘
```

### Workflow State Machine

```
Question Input
    │
    ├─→ [Analyze] Predict Difficulty & Extract Metrics
    │
    ├─→ [Retrieve] Query Pedagogical Knowledge Base
    │
    ├─→ [Reason] Generate Improvement Suggestions
    │
    ├─→ [Validate] Check Quality & Alignment
    │
    └─→ [Output] Structured Recommendations
```

## Key Features

### 1. Agentic Reasoning
- **State Management**: Explicit state tracking across workflow steps
- **Multi-step Reasoning**: Question analysis → Knowledge retrieval → Improvement generation
- **Autonomous Decision Making**: Agent decides which improvements to prioritize

### 2. RAG Integration
- **Pedagogical Knowledge Base**: Curated best practices for assessment design
- **Semantic Search**: Retrieve relevant guidance based on question context
- **Context-Aware Recommendations**: Suggestions tailored to question characteristics

### 3. Structured Output
- **Assessment Quality Score**: Overall quality metric (0-1)
- **Difficulty Distribution**: Breakdown of predicted difficulty levels
- **Learning Gap Analysis**: Identifies areas where students struggle
- **Actionable Improvements**: Prioritized suggestions with rationale

### 4. Responsible AI
- **Ethical Disclaimers**: Clear guidance that AI assists but doesn't replace educator judgment
- **Transparency**: Explains reasoning behind recommendations
- **Bias Awareness**: Acknowledges limitations of ML-based assessment

## Technical Stack

### Core Dependencies
```
langgraph>=0.0.1          # Agentic workflow orchestration
langchain>=0.1.0          # LLM framework
langchain-community       # Community integrations
chromadb>=0.4.0          # Vector database for RAG
ollama>=0.1.0            # Local LLM support
pydantic>=2.0.0          # Data validation
```

### Optional LLM Providers
- **Ollama** (Local, free): Recommended for deployment
- **OpenAI** (API-based): Higher quality but requires API key
- **Hugging Face** (Open-source): Community models

## Implementation Details

### State Definition
```python
class AssessmentState(BaseModel):
    question_text: str
    subject: str
    cognitive_level: str
    difficulty_prediction: Optional[str]
    confidence_scores: Optional[Dict[str, float]]
    analysis: Optional[str]
    improvements: Optional[list]
    pedagogical_context: Optional[str]
    final_recommendations: Optional[str]
```

### Agent Nodes

#### 1. Analysis Node
- Predicts question difficulty using trained ML model
- Extracts text metrics (word count, readability, etc.)
- Analyzes cognitive level alignment

#### 2. Retrieval Node
- Queries pedagogical knowledge base
- Retrieves relevant best practices
- Provides context for recommendations

#### 3. Reasoning Node
- Generates improvement suggestions
- Prioritizes by impact and feasibility
- Explains reasoning

#### 4. Validation Node
- Checks quality of recommendations
- Validates against pedagogical standards
- Ensures actionability

### RAG Knowledge Base

**Categories:**
1. **Bloom's Taxonomy**: Cognitive level guidance
2. **Question Design**: Best practices for clarity and alignment
3. **Difficulty Calibration**: Target success rates by level
4. **Learning Gaps**: Identifying and addressing knowledge gaps
5. **Assessment Quality**: Discrimination index and reliability

## Usage

### Running Milestone 2 App
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app_milestone2.py
```

### API Usage
```python
from app_milestone2 import AssessmentState, generate_improvements

# Create assessment state
state = AssessmentState(
    question_text="Solve the quadratic equation x² + 5x + 6 = 0",
    subject="Mathematics",
    cognitive_level="apply"
)

# Generate improvements
improvements = generate_improvements(state)
print(improvements)
```

## Evaluation Criteria

### Quality Metrics
- **Reasoning Quality**: Does the agent provide sound pedagogical reasoning?
- **RAG Effectiveness**: Are retrieved documents relevant and helpful?
- **Recommendation Utility**: Are suggestions actionable and valuable?
- **State Management**: Is workflow state properly tracked?

### Performance Benchmarks
- **Inference Time**: < 5 seconds per question
- **Accuracy**: Difficulty prediction > 35% (baseline: 33%)
- **Recommendation Coverage**: 80%+ of questions get suggestions

## Limitations & Future Work

### Current Limitations
1. **LLM Quality**: Depends on available open-source models
2. **Knowledge Base Size**: Limited to curated pedagogical content
3. **Context Window**: May miss nuanced question characteristics
4. **Hallucination Risk**: LLM may generate plausible but incorrect suggestions

### Future Enhancements
1. **Fine-tuned Models**: Train LLMs on educational data
2. **Expanded Knowledge Base**: Add domain-specific pedagogical content
3. **Multi-modal Analysis**: Support for diagrams, equations, code
4. **Feedback Loop**: Learn from educator corrections
5. **Collaborative Filtering**: Recommendations based on similar questions

## Deployment

### Streamlit Cloud
```bash
# Push to GitHub
git push origin main

# Deploy on Streamlit Cloud
# 1. Go to https://share.streamlit.io
# 2. Connect GitHub repo
# 3. Select app_milestone2.py
# 4. Deploy
```

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app_milestone2.py"]
```

### Environment Variables
```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=sk-...  # Optional
CHROMA_DB_PATH=./chroma_db
```

## References

### Pedagogical Frameworks
- Bloom's Taxonomy (Revised): Anderson & Krathwohl (2001)
- Assessment Design: Wiggins & McTighe (2005)
- Question Quality: Haladyna & Downing (1989)

### Technical References
- LangGraph: https://github.com/langchain-ai/langgraph
- Chroma: https://www.trychroma.com/
- Ollama: https://ollama.ai/

## Ethical Considerations

### Responsible AI Practices
1. **Transparency**: Clearly explain AI limitations
2. **Human-in-the-Loop**: Educators make final decisions
3. **Bias Mitigation**: Acknowledge ML model biases
4. **Privacy**: No student data collection
5. **Accessibility**: Ensure system is usable by all educators

### Disclaimers
- AI provides suggestions, not definitive assessments
- Educators should apply professional judgment
- System may have biases from training data
- Regular review and updates recommended

## Support & Contribution

### Getting Help
- GitHub Issues: Report bugs and feature requests
- Documentation: See README.md and docs/
- Community: Contribute improvements via pull requests

### Contributing
1. Fork the repository
2. Create feature branch
3. Submit pull request with documentation
4. Ensure tests pass

---

**Last Updated**: April 2026
**Status**: Production Ready
**Version**: 2.0.0
