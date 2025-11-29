"""
Enterprise Travel Assistant Streamlit Dashboard
Beautiful UI with chat interface and comprehensive metrics visualization
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🧳 Enterprise Travel Assistant",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful styling
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #3498db, #2980b9);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background: #f5f5f5;
        border-left: 4px solid #4caf50;
        margin-right: 2rem;
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_USER_ID = f"streamlit_user_{int(time.time())}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []


def check_api_health() -> Dict[str, Any]:
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def send_message(
    query: str, include_comparison: bool = True, use_cache: bool = True
) -> Dict[str, Any]:
    """Send message to travel assistant API"""
    try:
        payload = {
            "query": query,
            "user_id": st.session_state.user_id,
            "include_model_comparison": include_comparison,
            "use_cache": use_cache,
        }

        response = requests.post(
            f"{API_BASE_URL}/memory-travel-assistant", json=payload, timeout=30
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "query": query,
                "response": "Sorry, I'm experiencing technical difficulties.",
                "metrics": {},
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "response": "Unable to connect to the travel assistant service.",
            "metrics": {},
        }


def render_header():
    """Render the main header"""
    st.markdown(
        """
    <div class="main-header">
        <h1>🧳 Enterprise Travel Assistant Dashboard</h1>
        <p>AI-powered travel planning with advanced memory, caching & fingerprinting</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with controls and metrics"""
    st.sidebar.markdown("## 🎛️ Dashboard Controls")

    # API Health Check
    st.sidebar.markdown("### 🏥 System Health")
    if st.sidebar.button("Check API Health", use_container_width=True):
        health = check_api_health()
        if health.get("status") == "healthy":
            st.sidebar.markdown(
                '<p class="status-healthy">✅ System Healthy</p>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<p class="status-unhealthy">❌ System Issues</p>',
                unsafe_allow_html=True,
            )
            st.sidebar.error(health.get("error", "Unknown error"))

    # Chat Settings
    st.sidebar.markdown("### 💬 Chat Settings")

    # User ID
    new_user_id = st.sidebar.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Unique identifier for memory persistence",
    )
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.rerun()

    # Chat options
    include_comparison = st.sidebar.checkbox(
        "Include Model Comparison",
        value=True,
        help="Compare Gemini Flash vs Pro models",
    )

    use_cache = st.sidebar.checkbox(
        "Use Semantic Cache", value=True, help="Enable intelligent response caching"
    )

    # Clear chat button
    if st.sidebar.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Metrics refresh
    st.sidebar.markdown("### 📊 Metrics")
    if st.sidebar.button("Refresh Metrics", use_container_width=True):
        with st.spinner("Loading metrics..."):
            metrics = get_system_metrics()
            if "error" not in metrics:
                st.session_state.current_metrics = metrics
                st.session_state.metrics_timestamp = datetime.now()
        st.rerun()

    return include_comparison, use_cache


def render_chat_interface(include_comparison: bool, use_cache: bool):
    """Render the main chat interface"""
    st.markdown("## 💬 Travel Assistant Chat")

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        # Welcome message if no messages
        if not st.session_state.messages:
            st.markdown(
                """
            <div class="chat-message assistant-message">
                <strong>🤖 Travel Assistant:</strong><br>
                Hello! I'm your AI travel assistant with advanced memory capabilities. 
                I can help you plan trips, find destinations, and provide personalized recommendations. 
                I'll remember your preferences for future conversations!<br><br>
                What travel adventure are you planning? ✈️
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Display message history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>{message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Travel Assistant:</strong><br>{message["content"]}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Show metrics if available
                if "metrics" in message and message["metrics"]:
                    with st.expander("📊 Response Metrics", expanded=False):
                        render_message_metrics(message["metrics"])

    # Chat input
    st.markdown("### ✍️ Send a Message")

    # Create columns for input and button
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input(
            "Your travel question:",
            placeholder="e.g., I want to plan a romantic getaway to Europe...",
            label_visibility="collapsed",
            key="chat_input",
        )

    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")

    # Handle message sending
    if send_button and user_input.strip():
        # Add user message
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "timestamp": datetime.now()}
        )

        # Show thinking indicator
        with st.spinner(
            "🤖 Processing your request... (checking memory, cache, and consulting AI models)"
        ):
            # Send to API
            response = send_message(user_input, include_comparison, use_cache)

            # Add assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.get(
                        "response", "Sorry, I couldn't process your request."
                    ),
                    "timestamp": datetime.now(),
                    "metrics": response.get("metrics", {}),
                    "success": response.get("success", False),
                }
            )

            # Store metrics for history
            if response.get("metrics"):
                metrics_entry = {
                    "timestamp": datetime.now(),
                    "processing_time_ms": response.get("processing_time_ms", 0),
                    "success": response.get("success", False),
                    **response.get("metrics", {}),
                }
                st.session_state.metrics_history.append(metrics_entry)

                # Keep only last 50 entries
                if len(st.session_state.metrics_history) > 50:
                    st.session_state.metrics_history = st.session_state.metrics_history[
                        -50:
                    ]

        st.rerun()


def render_message_metrics(metrics: Dict[str, Any]):
    """Render metrics for a specific message"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Response Source", metrics.get("response_source", "Unknown"))
        cache_hit = metrics.get("cache_info", {}).get("cache_hit", False)
        st.metric("Cache Hit", "✅ Yes" if cache_hit else "❌ No")

    with col2:
        memories = metrics.get("memory_info", {}).get("memories_found", 0)
        st.metric("Memories Found", memories)
        duplicate = metrics.get("fingerprint_info", {}).get("is_duplicate", False)
        st.metric("Duplicate Request", "✅ Yes" if duplicate else "❌ No")

    with col3:
        if "model_comparison" in metrics:
            comparison = metrics["model_comparison"]
            st.metric("Flash Model Time", f"{comparison.get('flash_time_ms', 0):.0f}ms")
            st.metric("Pro Model Time", f"{comparison.get('pro_time_ms', 0):.0f}ms")


def render_system_metrics():
    """Render comprehensive system metrics"""
    st.markdown("## 📊 System Metrics & Analytics")

    # Check if we have current metrics
    if not hasattr(st.session_state, "current_metrics"):
        st.info("Click 'Refresh Metrics' in the sidebar to load system metrics.")
        return

    metrics = st.session_state.current_metrics

    # System Overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        system_metrics = metrics.get("system_metrics", {})
        st.metric("Total Requests", system_metrics.get("total_requests", 0))

    with col2:
        uptime_seconds = system_metrics.get("uptime_seconds", 0)
        uptime_str = str(timedelta(seconds=uptime_seconds))
        st.metric("System Uptime", uptime_str)

    with col3:
        avg_req_per_min = system_metrics.get("avg_requests_per_minute", 0)
        st.metric("Avg Requests/Min", f"{avg_req_per_min:.1f}")

    with col4:
        api_version = system_metrics.get("api_version", "Unknown")
        st.metric("API Version", api_version)

    # Component Metrics
    st.markdown("### 🧩 Component Performance")

    # Memory metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🧠 Memory System")
        memory_metrics = metrics.get("memory_metrics", {})

        if memory_metrics:
            memory_df = pd.DataFrame(
                [
                    {
                        "Metric": "Total Memories",
                        "Value": memory_metrics.get("total_memories", 0),
                    },
                    {
                        "Metric": "Avg Retrieval Time",
                        "Value": f"{memory_metrics.get('avg_retrieval_time_ms', 0):.1f}ms",
                    },
                    {
                        "Metric": "Memory Hits",
                        "Value": memory_metrics.get("memory_hits", 0),
                    },
                    {
                        "Metric": "Storage Ops",
                        "Value": memory_metrics.get("storage_operations", 0),
                    },
                ]
            )
            st.dataframe(memory_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 🗄️ Cache System")
        cache_metrics = metrics.get("cache_metrics", {})

        if cache_metrics:
            cache_df = pd.DataFrame(
                [
                    {
                        "Metric": "Cache Size",
                        "Value": cache_metrics.get("cache_size", 0),
                    },
                    {
                        "Metric": "Hit Rate",
                        "Value": f"{cache_metrics.get('hit_rate', 0):.1%}",
                    },
                    {
                        "Metric": "Total Hits",
                        "Value": cache_metrics.get("total_hits", 0),
                    },
                    {
                        "Metric": "Total Misses",
                        "Value": cache_metrics.get("total_misses", 0),
                    },
                ]
            )
            st.dataframe(cache_df, use_container_width=True, hide_index=True)

    # Model Performance
    st.markdown("#### 🤖 AI Model Performance")
    model_metrics = metrics.get("model_metrics", {})

    if model_metrics:
        col1, col2 = st.columns(2)

        with col1:
            # Flash model metrics
            flash_metrics = model_metrics.get("gemini_flash", {})
            st.metric("Flash Requests", flash_metrics.get("total_requests", 0))
            st.metric(
                "Flash Avg Time",
                f"{flash_metrics.get('avg_response_time_ms', 0):.1f}ms",
            )

        with col2:
            # Pro model metrics
            pro_metrics = model_metrics.get("gemini_pro", {})
            st.metric("Pro Requests", pro_metrics.get("total_requests", 0))
            st.metric(
                "Pro Avg Time", f"{pro_metrics.get('avg_response_time_ms', 0):.1f}ms"
            )

    # Performance Charts
    if (
        hasattr(st.session_state, "metrics_history")
        and st.session_state.metrics_history
    ):
        st.markdown("### 📈 Performance Trends")
        render_performance_charts()


def render_performance_charts():
    """Render performance visualization charts"""
    if not st.session_state.metrics_history:
        st.info(
            "No performance history available yet. Send some messages to see trends!"
        )
        return

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.metrics_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Processing time trend
    fig_time = px.line(
        df,
        x="timestamp",
        y="processing_time_ms",
        title="Response Processing Time Over Time",
        labels={"processing_time_ms": "Processing Time (ms)", "timestamp": "Time"},
    )
    fig_time.update_layout(height=300)
    st.plotly_chart(fig_time, use_container_width=True)

    # Success rate
    if len(df) > 1:
        df["success_rate"] = df["success"].rolling(window=min(10, len(df))).mean() * 100

        fig_success = px.line(
            df,
            x="timestamp",
            y="success_rate",
            title="Success Rate (Rolling 10-message Average)",
            labels={"success_rate": "Success Rate (%)", "timestamp": "Time"},
        )
        fig_success.update_layout(height=300)
        st.plotly_chart(fig_success, use_container_width=True)

    # Response source distribution
    if "response_source" in df.columns:
        source_counts = df["response_source"].value_counts()

        fig_pie = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Response Source Distribution",
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)


def main():
    """Main application function"""
    render_header()

    # Sidebar controls
    include_comparison, use_cache = render_sidebar()

    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 System Metrics"])

    with tab1:
        render_chat_interface(include_comparison, use_cache)

    with tab2:
        render_system_metrics()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        🧳 Enterprise Travel Assistant Dashboard | 
        Built with Streamlit, FastAPI & Google Gemini AI | 
        Features: Memory • Caching • Fingerprinting • Model Comparison
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
