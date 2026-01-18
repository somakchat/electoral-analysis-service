# Political Strategy Maker

**Advanced Multi-Agent Political Strategy System**

A sophisticated AI-powered political campaign strategy platform that uses 8 specialized agents working in a hierarchical crew architecture to provide micro-level constituency analysis and winning strategies.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategy Manager                          │
│              (Chief Political Strategist)                    │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  Research   │    │  Analysis   │    │  Strategy   │
    │    Team     │    │    Team     │    │    Team     │
    └─────────────┘    └─────────────┘    └─────────────┘
          │                   │                   │
    ┌─────┴─────┐       ┌─────┴─────┐       ┌─────┴─────┐
    │Intelligence│      │Data Science│      │  Ground   │
    │Opposition │       │Voter Analyst│     │ Resource  │
    │ Sentiment │       └───────────┘       └───────────┘
    └───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Strategic       │
                    │ Reporter        │
                    └─────────────────┘
```

## 🤖 8 Specialized Agents

| Agent | Specialization | Micro-Level Capabilities |
|-------|---------------|-------------------------|
| **Intelligence Agent** | Data Retrieval | Booth-level data, ward-wise patterns, historical trends |
| **Voter Analyst** | Demographic Analysis | Caste/community segments, age cohorts, occupation-based voting |
| **Opposition Research** | Competitor Analysis | Candidate strengths/weaknesses, anti-incumbency mapping |
| **Ground Strategist** | Field Operations | Rally locations, door-to-door coverage, influencer mapping |
| **Resource Optimizer** | Budget & Manpower | Fund allocation, volunteer deployment, media spend ROI |
| **Sentiment Decoder** | Opinion Analysis | Issue-wise sentiment, leader perception, grievances |
| **Data Scientist** | Statistical Analysis | Swing calculations, turnout modeling, vote transfer matrices |
| **Strategic Reporter** | Synthesis | Actionable briefs, risk alerts, strategy recommendations |

## 🔧 Advanced Features

### RAG Pipeline
- **Query Decomposition**: Breaks complex questions into searchable sub-queries
- **Hybrid Search**: Combines semantic (kNN) and keyword (BM25) search
- **Cross-Encoder Reranking**: Improves relevance with neural reranking
- **Contextual Compression**: Extracts only relevant evidence

### Decision Tools
- SWOT Analysis Tool
- Scenario Simulator Tool
- Resource Allocation Optimizer
- Micro-Targeting Tool

### Memory System
- Short-Term Memory (session context)
- Long-Term Memory (persistent learnings)
- Entity Memory (constituencies, candidates, parties)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (or Gemini API key)

### Local Development

1. **Clone and setup backend:**
```powershell
cd political-strategy-maker/backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. **Configure environment:**
```powershell
copy .env.template .env
# Edit .env with your API keys
```

3. **Start backend:**
```powershell
.\run_local.ps1
# Or: python -m uvicorn app.main:app --reload
```

4. **Setup frontend (new terminal):**
```powershell
cd political-strategy-maker/frontend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

5. **Start frontend:**
```powershell
.\run_frontend.ps1
# Or: streamlit run streamlit_app.py
```

6. **Access the application:**
- Frontend: http://localhost:8501
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws/chat

## 📊 API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload and index documents |
| GET | `/memory/{session_id}` | Get session history |
| GET | `/entities/{entity_type}` | Get entities by type |
| POST | `/chat` | Non-streaming chat (REST) |
| POST | `/quick-analysis` | Quick analysis mode |

### WebSocket API

Connect to `ws://host/ws/chat` and send:
```json
{
    "session_id": "uuid",
    "query": "Design a winning strategy for BJP in Nandigram",
    "constituency": "Nandigram",
    "party": "BJP"
}
```

Receive streaming updates:
```json
{"type": "agent_activity", "agent": "Intelligence Agent", "status": "working", "task": "..."}
{"type": "final_response", "answer": "...", "strategy": {...}, "citations": [...]}
```

## ☁️ AWS Deployment

### Prerequisites
- AWS SAM CLI
- AWS credentials configured

### Deploy
```bash
cd sam
sam build
sam deploy --guided
```

### Resources Created
- API Gateway (REST + WebSocket)
- Lambda Functions (5)
- DynamoDB Tables (2)
- S3 Bucket (1)

## 📁 Project Structure

```
political-strategy-maker/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration
│   │   ├── main.py             # FastAPI application
│   │   ├── models.py           # Pydantic models
│   │   ├── aws/                # Lambda handlers
│   │   │   ├── ws_connect.py
│   │   │   ├── ws_disconnect.py
│   │   │   ├── ws_chat.py
│   │   │   ├── ingest_handler.py
│   │   │   └── memory_handler.py
│   │   └── services/
│   │       ├── llm.py          # LLM providers
│   │       ├── orchestrator.py # Hierarchical crew
│   │       ├── ingest.py       # Document processing
│   │       ├── memory.py       # Memory system
│   │       ├── tools.py        # Decision tools
│   │       ├── agents/         # 8 specialist agents
│   │       │   ├── base.py
│   │       │   ├── intelligence.py
│   │       │   ├── voter_analyst.py
│   │       │   ├── opposition.py
│   │       │   ├── ground.py
│   │       │   ├── resource.py
│   │       │   ├── sentiment.py
│   │       │   ├── data_scientist.py
│   │       │   └── reporter.py
│   │       └── rag/            # Advanced RAG
│   │           ├── advanced_rag.py
│   │           ├── local_store.py
│   │           ├── opensearch_store.py
│   │           ├── embeddings.py
│   │           └── rerank.py
│   ├── requirements.txt
│   ├── .env.template
│   └── run_local.ps1
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── run_frontend.ps1
├── sam/
│   └── template.yaml
└── README.md
```

## 📝 Sample Usage

### Upload Documents
Upload electoral data documents (XLSX, DOCX, PDF) through the UI or API.

### Ask Strategy Questions
Examples:
- "Design a micro-level winning strategy for BJP in Nandigram constituency for 2026"
- "Analyze voter segments and identify persuadable groups in Diamond Harbour"
- "What are the key risks for TMC in North 24 Parganas?"
- "Optimize resource allocation across 10 priority constituencies"

### Get Comprehensive Analysis
The system provides:
- Executive summary
- SWOT analysis
- Voter segment analysis
- Ground operations plan
- Resource allocation recommendations
- Multiple election scenarios
- Priority actions
- Risk factors
- Success metrics

## 🔒 Security Notes

- Store API keys in environment variables or AWS Secrets Manager
- Use HTTPS in production
- Implement authentication for production deployments
- Review and restrict CORS settings

## 📜 License

Proprietary - For authorized use only.

## 👥 Support

For issues and feature requests, contact the development team.
