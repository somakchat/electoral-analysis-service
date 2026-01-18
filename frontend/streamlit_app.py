"""
Political Strategy Maker - Advanced Streamlit UI.

Features:
- Real-time agent activity visualization
- Document upload and ingestion
- Strategy chat interface
- Interactive dashboards
"""
import json
import uuid
import time
import threading
from queue import Queue, Empty
from datetime import datetime

import streamlit as st
import websocket
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ============= Page Configuration =============

st.set_page_config(
    page_title="Political Strategy Maker",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= Custom CSS =============

st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        margin: 0;
    }
    
    .main-header p {
        color: #eaeaea;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Agent cards */
    .agent-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .agent-card.working {
        border-color: #e94560;
        box-shadow: 0 0 15px rgba(233,69,96,0.3);
        animation: pulse 1.5s infinite;
    }
    
    .agent-card.done {
        border-color: #00d4aa;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 5px rgba(233,69,96,0.3); }
        50% { box-shadow: 0 0 20px rgba(233,69,96,0.5); }
        100% { box-shadow: 0 0 5px rgba(233,69,96,0.3); }
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
    }
    
    .assistant-message {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        border-left: 3px solid #e94560;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #e94560;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    
    /* Strategy sections */
    .strategy-section {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============= Session State Initialization =============

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_activities" not in st.session_state:
    st.session_state.agent_activities = {}

if "current_response" not in st.session_state:
    st.session_state.current_response = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ============= Sidebar Configuration =============

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Backend URLs
    BACKEND_HTTP = st.text_input(
        "Backend HTTP URL",
        value="http://127.0.0.1:8000",
        help="FastAPI backend URL"
    )
    BACKEND_WS = st.text_input(
        "Backend WebSocket URL", 
        value="ws://127.0.0.1:8000/ws/chat",
        help="WebSocket endpoint for real-time chat"
    )
    
    st.divider()
    
    # Session info
    st.markdown("### 📊 Session Info")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    st.caption(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.agent_activities = {}
        st.session_state.current_response = None
        st.rerun()

st.divider()

# Document upload
st.markdown("### 📁 Document Upload")
uploaded_files = st.file_uploader(
    "Upload documents for analysis",
    type=["xlsx", "xls", "xlsm", "docx", "pdf", "txt"],
    accept_multiple_files=True
)

if st.button("📥 Ingest Documents") and uploaded_files:
    with st.spinner("Ingesting documents..."):
        for f in uploaded_files:
            try:
                files = {"file": (f.name, f.getvalue())}
                response = requests.post(
                    f"{BACKEND_HTTP}/ingest",
                    files=files,
                    data={"extract_entities": "true"},
                    # Large TXT/PDF ingests can take several minutes (embedding + indexing)
                    timeout=600
                )
                if response.ok:
                    result = response.json()
                    st.success(f"✅ {f.name}: {result['chunks_indexed']} chunks, {result['entities_extracted']} entities")
                    status = result.get("index_status") or {}
                    if status:
                        st.caption(
                            " | ".join([
                                f"Local: {status.get('local_index', {}).get('status')} ({status.get('local_index', {}).get('chunks_indexed', 0)})",
                                f"OpenSearch: {status.get('opensearch', {}).get('status')} ({status.get('opensearch', {}).get('chunks_indexed', 0)})",
                                f"KG: {status.get('knowledge_graph', {}).get('status')} ({status.get('knowledge_graph', {}).get('facts_added', 0)})",
                            ])
                        )
                else:
                    st.error(f"❌ {f.name}: {response.text}")
            except Exception as e:
                st.error(f"❌ {f.name}: {str(e)}")

# ============= Main Content =============

# Header
st.markdown("""
<div class="main-header">
    <h1>🗳️ Political Strategy Maker</h1>
    <p>Advanced Multi-Agent Political Strategy System</p>
</div>
""", unsafe_allow_html=True)

# Main layout
col_chat, col_agents = st.columns([2, 1], gap="large")

# ============= Chat Column =============

with col_chat:
    st.markdown("### 💬 Strategy Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>Strategy AI:</strong><br>{msg["content"][:2000]}{'...' if len(msg["content"]) > 2000 else ''}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    
    col_input, col_button = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_area(
            "Your strategy question",
            placeholder="E.g., Design a micro-level winning strategy for BJP in Nandigram constituency for 2026 elections...",
            height=100,
            label_visibility="collapsed"
        )
    
    with col_button:
        send_button = st.button("🚀 Send", use_container_width=True, type="primary")
    
    # Optional parameters
    with st.expander("🎛️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            constituency = st.text_input("Constituency (optional)", placeholder="e.g., Nandigram")
        with col2:
            party = st.text_input("Party (optional)", placeholder="e.g., BJP")

# ============= Agent Activity Column =============

with col_agents:
    st.markdown("### 🤖 Agent Activity")
    
    agent_container = st.container()
    
    with agent_container:
        # Display agent status cards
        agents = [
            ("🔍", "Intelligence Agent", "intelligence"),
            ("📊", "Voter Analyst", "voter"),
            ("⚔️", "Opposition Research", "opposition"),
            ("🎯", "Ground Strategy", "ground"),
            ("💰", "Resource Optimizer", "resource"),
            ("💭", "Sentiment Decoder", "sentiment"),
            ("📈", "Data Scientist", "data"),
            ("📝", "Strategic Reporter", "reporter")
        ]
        
        for icon, name, key in agents:
            activity = st.session_state.agent_activities.get(name, {})
            status = activity.get("status", "idle")
            task = activity.get("task", "Waiting...")
            
            status_class = "working" if status == "working" else "done" if status == "done" else ""
            status_icon = "⏳" if status == "working" else "✅" if status == "done" else "💤"
            
            st.markdown(f"""
            <div class="agent-card {status_class}">
                <strong>{icon} {name}</strong> {status_icon}<br>
                <small style="color: #888;">{task[:50]}{'...' if len(task) > 50 else ''}</small>
            </div>
            """, unsafe_allow_html=True)

# ============= WebSocket Communication =============

def websocket_worker(query: str, session_id: str, constituency: str, party: str, output_queue: Queue):
    """Background worker for WebSocket communication."""
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            output_queue.put(data)
        except Exception as e:
            output_queue.put({"type": "error", "message": str(e)})

    def on_error(ws, error):
        output_queue.put({"type": "error", "message": str(error)})

    def on_close(ws, close_status_code, close_msg):
        output_queue.put({"type": "closed"})
    
    def on_open(ws):
        # Send the chat request
        request = {
            "session_id": session_id,
            "query": query,
            "workflow": "comprehensive_strategy",
            "depth": "micro",
            "include_scenarios": True
        }
        if constituency:
            request["constituency"] = constituency
        if party:
            request["party"] = party
        
        ws.send(json.dumps(request))

    ws = websocket.WebSocketApp(
        BACKEND_WS,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    ws.run_forever()

# ============= Process User Input =============

# Check for pending follow-up from interactive elements
if "pending_followup" in st.session_state and st.session_state.pending_followup:
    followup_query = st.session_state.pending_followup
    st.session_state.pending_followup = None  # Clear it
    
    # Process the follow-up automatically
    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": followup_query})
    st.session_state.agent_activities = {}
    
    # Create output queue and process
    output_queue = Queue()
    worker_thread = threading.Thread(
        target=websocket_worker,
        args=(followup_query, st.session_state.session_id, 
              st.session_state.get("constituency", ""), 
              st.session_state.get("party", ""), 
              output_queue),
        daemon=True
    )
    worker_thread.start()
    
    # Process responses (same as main query)
    progress_placeholder = st.empty()
    final_response = None
    start_time = time.time()
    
    with progress_placeholder.container():
        st.info(f"🔄 Processing follow-up: {followup_query[:50]}...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    while time.time() - start_time < 180:
        try:
            msg = output_queue.get(timeout=0.5)
            
            if msg.get("type") == "agent_activity":
                agent = msg.get("agent", "Unknown")
                st.session_state.agent_activities[agent] = {
                    "status": msg.get("status", "working"),
                    "task": msg.get("task", "")
                }
                done_count = sum(1 for a in st.session_state.agent_activities.values() 
                               if a.get("status") == "done")
                progress_bar.progress(min(0.9, done_count / 8))
                status_text.text(f"🤖 {agent}: {msg.get('task', '')[:50]}")
            
            elif msg.get("type") == "final_response":
                final_response = msg
                progress_bar.progress(1.0)
                break
            
            elif msg.get("type") == "error":
                st.error(f"Error: {msg.get('message', 'Unknown error')}")
                break
            
            elif msg.get("type") == "closed":
                break
        
        except Empty:
            continue
    
    progress_placeholder.empty()
    st.session_state.is_processing = False
    
    if final_response:
        answer = final_response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.current_response = final_response
        st.rerun()

# Process regular user input
if send_button and user_input and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.agent_activities = {}
    
    # Create output queue
    output_queue = Queue()
    
    # Start WebSocket worker
    worker_thread = threading.Thread(
        target=websocket_worker,
        args=(user_input, st.session_state.session_id, constituency, party, output_queue),
        daemon=True
    )
    worker_thread.start()
    
    # Process responses
    progress_placeholder = st.empty()
    final_response = None
    start_time = time.time()
    
    with progress_placeholder.container():
        st.info("🔄 Processing your request...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    while time.time() - start_time < 180:  # 3 minute timeout
        try:
            msg = output_queue.get(timeout=0.5)
            
            if msg.get("type") == "agent_activity":
                agent = msg.get("agent", "Unknown")
                st.session_state.agent_activities[agent] = {
                    "status": msg.get("status", "working"),
                    "task": msg.get("task", "")
                }
                
                # Update progress
                done_count = sum(1 for a in st.session_state.agent_activities.values() 
                               if a.get("status") == "done")
                progress_bar.progress(min(0.9, done_count / 8))
                status_text.text(f"🤖 {agent}: {msg.get('task', '')[:50]}")
            
            elif msg.get("type") == "final_response":
                final_response = msg
                progress_bar.progress(1.0)
                break
            
            elif msg.get("type") == "error":
                st.error(f"Error: {msg.get('message', 'Unknown error')}")
                break
            
            elif msg.get("type") == "closed":
                break
        
        except Empty:
            continue
    
    progress_placeholder.empty()
    st.session_state.is_processing = False
    
    if final_response:
        answer = final_response.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.current_response = final_response
        st.rerun()
    else:
        st.error("Request timed out. Please try again.")

# ============= Display Strategy Results =============

if st.session_state.current_response:
    response = st.session_state.current_response
    
    st.markdown("---")
    st.markdown("### 📋 Strategy Analysis")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Agents Used</div>
        </div>
        """.format(len(response.get("agents_used", []))), unsafe_allow_html=True)
    
    with col2:
        confidence = response.get("confidence", 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{confidence:.0f}%</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        citations = len(response.get("citations", []))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{citations}</div>
            <div class="metric-label">Citations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        stored = "✅" if response.get("memory_stored") else "❌"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stored}</div>
            <div class="metric-label">Memory Stored</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategy details
    strategy = response.get("strategy", {})
    
    if strategy:
        # Executive Summary
        if strategy.get("executive_summary"):
            with st.expander("📝 Executive Summary", expanded=True):
                st.markdown(strategy["executive_summary"])
        
        # SWOT Analysis
        if strategy.get("swot_analysis"):
            with st.expander("💪 SWOT Analysis"):
                swot = strategy["swot_analysis"]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Strengths** 💪")
                    for s in swot.get("strengths", []):
                        st.markdown(f"- {s}")
                    
                    st.markdown("**Opportunities** 🎯")
                    for o in swot.get("opportunities", []):
                        st.markdown(f"- {o}")
                
                with col2:
                    st.markdown("**Weaknesses** ⚠️")
                    for w in swot.get("weaknesses", []):
                        st.markdown(f"- {w}")
                    
                    st.markdown("**Threats** 🚨")
                    for t in swot.get("threats", []):
                        st.markdown(f"- {t}")
        
        # Voter Segments
        if strategy.get("voter_segments"):
            with st.expander("👥 Voter Segments"):
                segments = strategy["voter_segments"]
                for seg in segments[:5]:
                    st.markdown(f"""
                    **{seg.get('segment_name', 'Unknown')}** 
                    - Share: {seg.get('population_share', 'N/A')}%
                    - Support: {seg.get('current_support', 'N/A')}
                    - Persuadability: {seg.get('persuadability', 'N/A')}
                    - Strategy: {seg.get('strategy', 'N/A')}
                    """)
                    st.divider()
        
        # Scenarios
        if strategy.get("scenarios"):
            with st.expander("🎲 Election Scenarios"):
                scenarios = strategy["scenarios"]
                
                # Create scenario comparison chart
                if scenarios:
                    fig = go.Figure()
                    
                    for scenario in scenarios:
                        name = scenario.get("name", "Unknown")
                        vote_share = scenario.get("projected_vote_share", "0%")
                        # Extract number from percentage string
                        try:
                            share_value = float(vote_share.replace("%", ""))
                        except:
                            share_value = 0
                        
                        fig.add_trace(go.Bar(
                            name=name,
                            x=[name],
                            y=[share_value],
                            text=[f"{vote_share}<br>{scenario.get('outcome', '')}"],
                            textposition='outside'
                        ))
                    
                    fig.update_layout(
                        title="Scenario Projections",
                        yaxis_title="Projected Vote Share (%)",
                        template="plotly_dark",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Priority Actions
        if strategy.get("priority_actions"):
            with st.expander("🎯 Priority Actions"):
                for i, action in enumerate(strategy["priority_actions"], 1):
                    st.markdown(f"{i}. {action}")
        
        # Risk Factors
        if strategy.get("risk_factors"):
            with st.expander("⚠️ Risk Factors"):
                for risk in strategy["risk_factors"]:
                    st.warning(risk)
    
    # Citations
    citations = response.get("citations", [])
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})"):
            for i, cite in enumerate(citations[:10], 1):
                # Handle None values safely
                chunk_id = cite.get('chunk_id') or cite.get('doc_id') or 'N/A'
                source = cite.get('source_path') or cite.get('source') or 'N/A'
                score = cite.get('score') or cite.get('relevance_score') or 0
                text = cite.get('text') or cite.get('content') or ''
                
                st.markdown(f"""
                **[{i}]** `{chunk_id}`  
                Source: {source}  
                Score: {float(score):.3f}
                
                > {text[:300]}{'...' if len(text) > 300 else ''}
                """)
                st.divider()
    
    # ============= Interactive Follow-ups (Human-in-the-Loop) =============
    interactions = response.get("interactions", [])
    if interactions:
        st.markdown("---")
        st.markdown("### 🔄 Continue the Conversation")
        
        for interaction in interactions:
            interaction_type = interaction.get("type", "")
            message = interaction.get("message", "")
            options = interaction.get("options", [])
            priority = interaction.get("priority", "suggested")
            
            # Style based on interaction type
            if interaction_type == "follow_up":
                st.info(f"💡 **{message}**")
                
                # Display options as columns of buttons
                if options:
                    cols = st.columns(min(len(options), 4))
                    for i, opt in enumerate(options[:4]):
                        with cols[i]:
                            if st.button(
                                opt.get("label", "Option"),
                                key=f"followup_{opt.get('id', i)}",
                                use_container_width=True
                            ):
                                # Store selected follow-up in session state
                                # Follow-ups should send the *actual query text* (label), not the id
                                st.session_state.pending_followup = opt.get("label") or opt.get("id")
                                st.rerun()
            
            elif interaction_type == "clarification":
                if priority == "blocking":
                    st.warning(f"❓ **{message}**")
                else:
                    st.info(f"❓ {message}")
                
                # Display clarification options
                if options:
                    cols = st.columns(min(len(options), 3))
                    for i, opt in enumerate(options[:3]):
                        with cols[i]:
                            if st.button(
                                opt.get("label", "Option"),
                                key=f"clarify_{opt.get('id', i)}",
                                type="primary" if i == 0 else "secondary",
                                use_container_width=True
                            ):
                                # Set the clarification response
                                st.session_state.pending_followup = opt.get("id") or opt.get("label")
                                st.rerun()
            
            elif interaction_type == "refinement":
                with st.expander("✏️ Refine this response"):
                    st.caption(message)
                    cols = st.columns(3)
                    for i, opt in enumerate(options[:6]):
                        col_idx = i % 3
                        with cols[col_idx]:
                            if st.button(
                                opt.get("label", "Refine"),
                                key=f"refine_{opt.get('id', i)}",
                                use_container_width=True
                            ):
                                st.session_state.refinement_request = opt.get("id")
                                st.session_state.pending_followup = f"Please {opt.get('label', '').lower()}"
                                st.rerun()
            
            elif interaction_type == "confirmation":
                st.warning(f"⚠️ **Confirmation Required:** {message}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Confirm", key="confirm_yes", type="primary"):
                        st.session_state.confirmation_response = "confirmed"
                        st.success("Confirmed!")
                with col2:
                    if st.button("✏️ Modify", key="confirm_modify"):
                        st.session_state.confirmation_response = "modify"
                with col3:
                    if st.button("❌ Cancel", key="confirm_cancel"):
                        st.session_state.confirmation_response = "cancelled"
                        st.info("Action cancelled.")
    
    # ============= Quick Actions =============
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    quick_cols = st.columns(4)
    
    quick_actions = [
        ("🎯 Swing Seats", "What are the swing seats in West Bengal?"),
        ("📊 TMC Analysis", "What is TMC's current position?"),
        ("📈 BJP Strategy", "What strategic actions should BJP take?"),
        ("🗺️ District View", "Analyze the key districts")
    ]
    
    for i, (label, query) in enumerate(quick_actions):
        with quick_cols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_followup = query
                st.rerun()
    
    # ============= Feedback Section =============
    st.markdown("---")
    st.markdown("### 💬 Provide Feedback")
    st.caption("Help us improve by providing feedback on this response")
    
    response_id = response.get("response_id", f"{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    
    col_rating, col_feedback = st.columns([1, 2])
    
    with col_rating:
        st.markdown("#### Rate this response")
        rating = st.slider("Rating", 1, 5, 3, key="feedback_rating",
                          help="1 = Poor, 5 = Excellent")
        
        rating_emojis = {1: "😞", 2: "😐", 3: "🙂", 4: "😊", 5: "🤩"}
        st.markdown(f"<h2 style='text-align:center'>{rating_emojis[rating]}</h2>", unsafe_allow_html=True)
        
        if st.button("📤 Submit Rating", key="submit_rating"):
            try:
                response_data = requests.post(
                    f"{BACKEND_HTTP}/feedback",
                    json={
                        "session_id": st.session_state.session_id,
                        "response_id": response_id,
                        "feedback_type": "rating",
                        "rating": rating
                    },
                    timeout=10
                )
                if response_data.ok:
                    st.success("✅ Rating submitted! Thank you!")
                else:
                    st.error(f"Error: {response_data.text}")
            except Exception as e:
                st.error(f"Failed to submit: {str(e)}")
    
    with col_feedback:
        st.markdown("#### Correction or Suggestion")
        
        feedback_type = st.selectbox(
            "Feedback type",
            ["correction", "addition", "disagreement", "comment"],
            help="What kind of feedback are you providing?"
        )
        
        if feedback_type == "correction":
            original_text = st.text_input("What was incorrect?", 
                                          placeholder="e.g., 'BJP won 8 seats'",
                                          key="original_text")
            corrected_text = st.text_input("What is the correct information?",
                                           placeholder="e.g., 'BJP won 9 seats'",
                                           key="corrected_text")
            entity_name = st.text_input("Entity name (constituency/party)",
                                        placeholder="e.g., BANKURA",
                                        key="entity_name")
        elif feedback_type == "addition":
            corrected_text = st.text_area("New information to add",
                                          placeholder="Enter additional information...",
                                          key="addition_text")
            entity_name = st.text_input("Related entity (optional)",
                                        placeholder="e.g., NANDIGRAM",
                                        key="entity_name_add")
            original_text = ""
        else:
            comment_text = st.text_area("Your feedback",
                                        placeholder="Share your thoughts...",
                                        key="comment_text")
            original_text = ""
            corrected_text = comment_text if feedback_type == "disagreement" else ""
            entity_name = ""
        
        if st.button("📤 Submit Feedback", key="submit_feedback"):
            try:
                feedback_data = {
                    "session_id": st.session_state.session_id,
                    "response_id": response_id,
                    "feedback_type": feedback_type
                }
                
                if feedback_type == "correction":
                    feedback_data["original_text"] = original_text
                    feedback_data["corrected_text"] = corrected_text
                    feedback_data["entity_name"] = entity_name
                elif feedback_type == "addition":
                    feedback_data["corrected_text"] = corrected_text
                    feedback_data["entity_name"] = entity_name
                else:
                    feedback_data["comment"] = comment_text if 'comment_text' in dir() else corrected_text
                
                response_data = requests.post(
                    f"{BACKEND_HTTP}/feedback",
                    json=feedback_data,
                    timeout=10
                )
                if response_data.ok:
                    result = response_data.json()
                    if result.get("correction_applied"):
                        st.success("✅ Correction applied! Future responses will be improved.")
                    elif result.get("learning_updated"):
                        st.success("✅ Information added to knowledge base!")
                    else:
                        st.success(f"✅ {result.get('message', 'Feedback recorded!')}")
                else:
                    st.error(f"Error: {response_data.text}")
            except Exception as e:
                st.error(f"Failed to submit: {str(e)}")

# ============= Feedback Stats in Sidebar =============

with st.sidebar:
    st.divider()
    st.markdown("### 📈 Learning Stats")
    
    try:
        stats_response = requests.get(f"{BACKEND_HTTP}/feedback/stats", timeout=5)
        if stats_response.ok:
            stats = stats_response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Feedback", stats.get("total_feedback", 0))
            with col2:
                st.metric("Corrections", stats.get("total_corrections", 0))
            
            avg_rating = stats.get("average_rating", 0)
            if avg_rating > 0:
                st.caption(f"Avg Rating: {'⭐' * int(avg_rating)} ({avg_rating:.1f}/5)")
    except:
        st.caption("Stats unavailable")

# ============= Footer =============

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Political Strategy Maker v1.0 | Powered by Multi-Agent AI</small>
</div>
""", unsafe_allow_html=True)
