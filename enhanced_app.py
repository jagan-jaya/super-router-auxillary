"""
Enhanced Routing Optimization System - Real-time Web UI
A comprehensive web application for generating, viewing, and analyzing realistic transaction data in real-time.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import io
from datetime import datetime, timedelta
import sys
import os
import time
import threading
import queue
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.realistic_transaction_generator import RealisticTransactionGenerator
from src.data_generation.enhanced_realistic_generator import EnhancedRealisticTransactionGenerator, TimeRange
from src.models import TransactionStatus

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Routing Optimization System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = []
if 'real_time_running' not in st.session_state:
    st.session_state.real_time_running = False
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .failure-metric {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .cost-metric {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    .volume-metric {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #333;
    }
    .real-time-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #28a745;
        border-radius: 50%;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Sidebar navigation buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar metrics styling */
    .css-1d391kg .metric {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }
    
    /* Page title styling */
    h1, h2, h3 {
        color: #2d3748;
    }
    
    /* Enhanced button styling for main content */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class RealTimeDataGenerator:
    """Real-time data generator for streaming transactions."""
    
    def __init__(self, merchants: List[str], seed: int = 42):
        self.generator = RealisticTransactionGenerator(seed=seed)
        self.merchants = merchants
        self.running = False
        self.data_queue = queue.Queue()
        
    def start_generation(self, transactions_per_minute: int = 10):
        """Start generating real-time data."""
        self.running = True
        self.transactions_per_minute = transactions_per_minute
        thread = threading.Thread(target=self._generate_data)
        thread.daemon = True
        thread.start()
        
    def stop_generation(self):
        """Stop generating real-time data."""
        self.running = False
        
    def _generate_data(self):
        """Generate data in background thread."""
        while self.running:
            try:
                # Generate transactions for random merchants
                merchant_id = np.random.choice(self.merchants)
                transactions = self.generator.generate_realistic_dataset(
                    merchant_id, 
                    num_transactions=1
                )
                
                for transaction in transactions:
                    self.data_queue.put(transaction.model_dump())
                
                # Wait based on transactions per minute
                wait_time = 60 / self.transactions_per_minute
                time.sleep(wait_time)
                
            except Exception as e:
                st.error(f"Error in real-time generation: {e}")
                break
                
    def get_new_data(self) -> List[Dict]:
        """Get new data from queue."""
        new_data = []
        while not self.data_queue.empty():
            try:
                new_data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return new_data

def load_data():
    """Load existing transaction data if available."""
    try:
        if os.path.exists('data/realistic_all_transactions.csv'):
            return pd.read_csv('data/realistic_all_transactions.csv')
        else:
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def generate_batch_data(num_merchants, transactions_per_merchant, seed):
    """Generate batch transaction data."""
    with st.spinner("Generating realistic transaction data..."):
        generator = RealisticTransactionGenerator(seed=seed)
        all_transactions = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_merchants):
            merchant_id = f"merchant_{i+1:03d}"
            status_text.text(f"Generating data for {merchant_id}...")
            
            transactions = generator.generate_realistic_dataset(
                merchant_id, transactions_per_merchant
            )
            all_transactions.extend(transactions)
            
            progress_bar.progress((i + 1) / num_merchants)
        
        # Convert to DataFrame
        df = pd.DataFrame([t.model_dump() for t in all_transactions])
        
        # Save to file
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/realistic_all_transactions.csv', index=False)
        
        status_text.text("✅ Data generation completed!")
        progress_bar.progress(1.0)
        
        return df

def create_enhanced_metrics(df):
    """Create enhanced overview metrics cards."""
    total_transactions = len(df)
    
    # Handle multiple success statuses
    success_statuses = ['success', 'authorized', 'captured']
    successful_transactions = len(df[df['status'].isin(success_statuses)])
    
    # Handle multiple failure statuses
    failure_statuses = ['failed', 'declined', 'timeout', 'cancelled', 'expired', 'voided']
    failed_transactions = len(df[df['status'].isin(failure_statuses)])
    
    success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    total_volume = df['amount'].sum()
    
    # Calculate cost for successful transactions only
    successful_df = df[df['status'].isin(success_statuses)]
    total_cost = successful_df['cost'].sum() if len(successful_df) > 0 else 0
    
    avg_processing_time = df['processing_time_ms'].mean()
    
    # Count unique statuses
    unique_statuses = df['status'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card volume-metric">
            <h3>📊 Total Transactions</h3>
            <h1>{total_transactions:,}</h1>
            <p>Across {df['merchant_id'].nunique()} merchants</p>
            <small>Last updated: {datetime.now().strftime('%H:%M:%S')}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>✅ Success Rate</h3>
            <h1>{success_rate:.1f}%</h1>
            <p>{successful_transactions:,} successful transactions</p>
            <small>{unique_statuses} unique statuses</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card volume-metric">
            <h3>💰 Total Volume</h3>
            <h1>${total_volume:,.0f}</h1>
            <p>Processing cost: ${total_cost:,.2f}</p>
            <small>Cost ratio: {(total_cost/total_volume)*100:.3f}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card cost-metric">
            <h3>⚡ Avg Processing Time</h3>
            <h1>{avg_processing_time:.0f}ms</h1>
            <p>Fastest: {df['processing_time_ms'].min():.0f}ms</p>
            <small>Slowest: {df['processing_time_ms'].max():.0f}ms</small>
        </div>
        """, unsafe_allow_html=True)

def create_real_time_chart(df):
    """Create real-time transaction flow chart."""
    if len(df) == 0:
        return None
        
    # Get last 100 transactions for real-time view
    recent_df = df.tail(100).copy()
    recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp'])
    recent_df = recent_df.sort_values('timestamp')
    
    # Create cumulative success/failure counts
    recent_df['cumulative_success'] = (recent_df['status'] == 'success').cumsum()
    recent_df['cumulative_failure'] = (recent_df['status'] == 'failed').cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['cumulative_success'],
        mode='lines+markers',
        name='Successful Transactions',
        line=dict(color='#28a745', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['cumulative_failure'],
        mode='lines+markers',
        name='Failed Transactions',
        line=dict(color='#dc3545', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Real-time Transaction Flow (Last 100 Transactions)",
        xaxis_title="Time",
        yaxis_title="Cumulative Count",
        hovermode='x unified',
        height=400,
        showlegend=True
    )
    
    return fig

def create_connector_heatmap(df):
    """Create connector performance heatmap."""
    # Create pivot table for heatmap
    pivot_data = df.groupby(['connector_id', 'payment_method']).agg({
        'status': lambda x: (x == 'success').mean() * 100
    }).reset_index()
    
    pivot_table = pivot_data.pivot(index='connector_id', columns='payment_method', values='status')
    
    fig = px.imshow(
        pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale='RdYlGn',
        title="Connector Success Rate Heatmap by Payment Method (%)",
        labels=dict(color="Success Rate (%)")
    )
    
    fig.update_layout(height=500)
    return fig

def create_advanced_analytics(df):
    """Create advanced analytics charts."""
    
    # 1. Transaction Volume by Hour and Day
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.day_name()
    
    hourly_volume = df_copy.groupby('hour')['amount'].sum().reset_index()
    daily_volume = df_copy.groupby('day_of_week')['amount'].sum().reset_index()
    
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Hourly Transaction Volume', 'Daily Transaction Volume'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig1.add_trace(
        go.Bar(x=hourly_volume['hour'], y=hourly_volume['amount'], name='Hourly'),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Bar(x=daily_volume['day_of_week'], y=daily_volume['amount'], name='Daily'),
        row=1, col=2
    )
    
    fig1.update_layout(height=400, showlegend=False, title_text="Transaction Volume Patterns")
    
    # 2. Risk Score Distribution
    fig2 = px.histogram(
        df, x='risk_score', nbins=20,
        title="Risk Score Distribution",
        labels={'risk_score': 'Risk Score', 'count': 'Number of Transactions'},
        color_discrete_sequence=['#1f77b4']
    )
    fig2.update_layout(height=400)
    
    # 3. Amount vs Processing Time Scatter
    fig3 = px.scatter(
        df.sample(min(1000, len(df))),  # Sample for performance
        x='amount', y='processing_time_ms',
        color='status',
        title="Transaction Amount vs Processing Time",
        labels={'amount': 'Transaction Amount ($)', 'processing_time_ms': 'Processing Time (ms)'},
        color_discrete_map={'success': '#28a745', 'failed': '#dc3545'}
    )
    fig3.update_layout(height=400)
    
    # 4. Connector-wise Transaction Volume Over Time
    df_copy['date'] = df_copy['timestamp'].dt.date
    connector_volume_time = df_copy.groupby(['date', 'connector_id'])['amount'].sum().reset_index()
    
    fig4 = px.line(
        connector_volume_time,
        x='date',
        y='amount',
        color='connector_id',
        title="Connector-wise Transaction Volume Over Time",
        labels={'amount': 'Transaction Volume ($)', 'date': 'Date', 'connector_id': 'Connector'},
        markers=True
    )
    fig4.update_layout(height=500, hovermode='x unified')
    
    # 5. Connector Success Rate Over Time
    # Calculate success rate per connector per day
    success_statuses = ['success', 'authorized', 'captured']
    df_copy['is_successful'] = df_copy['status'].isin(success_statuses)
    
    connector_success_time = df_copy.groupby(['date', 'connector_id']).agg({
        'is_successful': ['sum', 'count']
    }).reset_index()
    
    # Flatten column names
    connector_success_time.columns = ['date', 'connector_id', 'successful_count', 'total_count']
    connector_success_time['success_rate'] = (connector_success_time['successful_count'] / 
                                            connector_success_time['total_count']) * 100
    
    fig5 = px.line(
        connector_success_time,
        x='date',
        y='success_rate',
        color='connector_id',
        title="Connector Success Rate Over Time",
        labels={'success_rate': 'Success Rate (%)', 'date': 'Date', 'connector_id': 'Connector'},
        markers=True
    )
    fig5.update_layout(height=500, hovermode='x unified', yaxis=dict(range=[0, 100]))
    
    return fig1, fig2, fig3, fig4, fig5

def create_ml_insights(df):
    """Create machine learning insights."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Prepare data for ML
    ml_df = df.copy()
    
    # Encode categorical variables
    le_merchant = LabelEncoder()
    le_connector = LabelEncoder()
    le_payment_method = LabelEncoder()
    le_country = LabelEncoder()
    
    ml_df['merchant_encoded'] = le_merchant.fit_transform(ml_df['merchant_id'])
    ml_df['connector_encoded'] = le_connector.fit_transform(ml_df['connector_id'])
    ml_df['payment_method_encoded'] = le_payment_method.fit_transform(ml_df['payment_method'])
    ml_df['country_encoded'] = le_country.fit_transform(ml_df['country_code'])
    
    # Features for prediction
    features = ['amount', 'risk_score', 'merchant_encoded', 'connector_encoded', 
                'payment_method_encoded', 'country_encoded']
    
    X = ml_df[features]
    y = (ml_df['status'] == 'success').astype(int)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create feature importance chart
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance for Transaction Success Prediction",
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    fig.update_layout(height=400)
    
    # Model performance
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    
    return fig, accuracy, feature_importance

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🚀 Enhanced Routing Optimization System</h1>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        # Navigation buttons
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.selected_page = "📊 Dashboard"
        if st.button("⚡ Real-time Monitor", use_container_width=True):
            st.session_state.selected_page = "⚡ Real-time Monitor"
        if st.button("🔍 Advanced Analytics", use_container_width=True):
            st.session_state.selected_page = "🔍 Advanced Analytics"
        if st.button("🤖 ML Insights", use_container_width=True):
            st.session_state.selected_page = "🤖 ML Insights"
        if st.button("🛠️ Data Management", use_container_width=True):
            st.session_state.selected_page = "🛠️ Data Management"
        
        # Initialize selected page if not set
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "📊 Dashboard"
        
        # Show current selection
        st.markdown("---")
        st.markdown(f"**Current Page:** {st.session_state.selected_page}")
        
        # Quick stats in sidebar
        df = load_data()
        if df is not None:
            st.markdown("---")
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Transactions", f"{len(df):,}")
            st.metric("Success Rate", f"{(df['status'].isin(['success', 'authorized', 'captured'])).mean():.1%}")
            st.metric("Merchants", df['merchant_id'].nunique())
            st.metric("Connectors", df['connector_id'].nunique())
    
    # Get selected page
    selected = st.session_state.selected_page
    
    # Load existing data
    df = load_data()
    
    if selected == "📊 Dashboard":
        if df is None:
            st.warning("⚠️ No transaction data found. Please generate data in the Data Management section.")
            return
            
        st.header("📊 Transaction Dashboard")
        
        # Enhanced metrics
        create_enhanced_metrics(df)
        
        st.markdown("---")
        
        # Charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution
            status_counts = df['status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Transaction Status Distribution",
                color_discrete_map={'success': '#28a745', 'failed': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Geographic distribution
            country_stats = df['country_code'].value_counts().head(10)
            fig = px.bar(
                x=country_stats.index,
                y=country_stats.values,
                title="Geographic Distribution (Top 10 Countries)",
                labels={'x': 'Country Code', 'y': 'Number of Transactions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Real-time chart
        real_time_fig = create_real_time_chart(df)
        if real_time_fig:
            st.plotly_chart(real_time_fig, use_container_width=True)
        
        # Connector heatmap
        st.plotly_chart(create_connector_heatmap(df), use_container_width=True)
    
    elif selected == "⚡ Real-time Monitor":
        st.header("⚡ Real-time Transaction Monitor")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.real_time_running:
                st.markdown('<span class="real-time-indicator"></span>**Real-time generation is ACTIVE**', unsafe_allow_html=True)
            else:
                st.info("Real-time generation is stopped")
        
        with col2:
            transactions_per_minute = st.slider("Transactions/min", 1, 60, 10)
        
        with col3:
            if not st.session_state.real_time_running:
                if st.button("🚀 Start Real-time", type="primary"):
                    merchants = [f"merchant_{i:03d}" for i in range(1, 4)]
                    st.session_state.generator = RealTimeDataGenerator(merchants)
                    st.session_state.generator.start_generation(transactions_per_minute)
                    st.session_state.real_time_running = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Real-time", type="secondary"):
                    if st.session_state.generator:
                        st.session_state.generator.stop_generation()
                    st.session_state.real_time_running = False
                    st.rerun()
        
        # Auto-refresh for real-time updates
        if st.session_state.real_time_running:
            time.sleep(2)  # Refresh every 2 seconds
            
            # Get new data
            if st.session_state.generator:
                new_data = st.session_state.generator.get_new_data()
                if new_data:
                    st.session_state.real_time_data.extend(new_data)
                    
                    # Keep only last 1000 transactions for performance
                    if len(st.session_state.real_time_data) > 1000:
                        st.session_state.real_time_data = st.session_state.real_time_data[-1000:]
            
            st.rerun()
        
        # Display real-time data
        if st.session_state.real_time_data:
            rt_df = pd.DataFrame(st.session_state.real_time_data)
            
            # Real-time metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Live Transactions", len(rt_df))
            with col2:
                success_rate = (rt_df['status'] == 'success').mean() * 100
                st.metric("Live Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Live Volume", f"${rt_df['amount'].sum():,.0f}")
            with col4:
                avg_time = rt_df['processing_time_ms'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.0f}ms")
            
            # Real-time chart
            real_time_fig = create_real_time_chart(rt_df)
            if real_time_fig:
                st.plotly_chart(real_time_fig, use_container_width=True)
            
            # Recent transactions table
            st.subheader("Recent Transactions")
            st.dataframe(rt_df.tail(20), use_container_width=True)
        else:
            st.info("No real-time data available. Start real-time generation to see live transactions.")
    
    elif selected == "🔍 Advanced Analytics":
        if df is None:
            st.warning("⚠️ No transaction data found. Please generate data in the Data Management section.")
            return
            
        st.header("🔍 Advanced Analytics")
        
        # Advanced analytics charts
        fig1, fig2, fig3, fig4, fig5 = create_advanced_analytics(df)
        
        # Basic analytics
        st.plotly_chart(fig1, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
        
        # Connector Time Series Analytics
        st.markdown("---")
        st.subheader("🔗 Connector Performance Over Time")
        
        # Connector volume over time
        st.plotly_chart(fig4, use_container_width=True)
        
        # Connector success rate over time
        st.plotly_chart(fig5, use_container_width=True)
        
        # Detailed statistics
        st.markdown("---")
        st.subheader("📈 Statistical Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Transaction Amount Statistics**")
            st.write(df['amount'].describe())
            
        with col2:
            st.write("**Processing Time Statistics**")
            st.write(df['processing_time_ms'].describe())
        
        with col3:
            st.write("**Connector Performance Summary**")
            connector_summary = df.groupby('connector_id').agg({
                'amount': ['count', 'sum', 'mean'],
                'status': lambda x: (x.isin(['success', 'authorized', 'captured'])).mean()
            }).round(3)
            connector_summary.columns = ['Transactions', 'Total Volume', 'Avg Amount', 'Success Rate']
            st.write(connector_summary)
    
    elif selected == "🤖 ML Insights":
        if df is None:
            st.warning("⚠️ No transaction data found. Please generate data in the Data Management section.")
            return
            
        st.header("🤖 Machine Learning Insights")
        
        with st.spinner("Training ML model..."):
            try:
                fig, accuracy, feature_importance = create_ml_insights(df)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                    st.write("**Top Features:**")
                    for _, row in feature_importance.head(3).iterrows():
                        st.write(f"• {row['feature']}: {row['importance']:.3f}")
                
                # Feature importance table
                st.subheader("📊 Feature Importance Details")
                st.dataframe(feature_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in ML analysis: {e}")
    
    elif selected == "🛠️ Data Management":
        st.header("🛠️ Data Management")
        
        tab1, tab2, tab3 = st.tabs(["📊 Generate Data", "📁 Current Data", "⚙️ Settings"])
        
        with tab1:
            st.subheader("🚀 Enhanced Data Generation")
            
            # Generator selection
            generator_type = st.radio(
                "Select Generator Type:",
                ["🔥 Enhanced Generator (15+ statuses, downtime simulation)", "📊 Basic Generator (legacy)"],
                index=0
            )
            
            if generator_type.startswith("🔥"):
                st.info("✨ Enhanced generator includes: 15+ transaction statuses, artificial downtime, time-based patterns, and realistic failure scenarios")
                
                # Enhanced generator options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📅 Time Range Configuration")
                    time_range = st.selectbox(
                        "Time Range",
                        ["hours", "days", "weeks", "months", "years"],
                        index=1,
                        help="Generate data spanning different time periods"
                    )
                    
                    duration = st.number_input(
                        f"Duration ({time_range})",
                        min_value=1,
                        max_value=365 if time_range == "days" else (52 if time_range == "weeks" else (12 if time_range == "months" else (5 if time_range == "years" else 168))),
                        value=7 if time_range == "days" else (2 if time_range == "weeks" else (1 if time_range in ["months", "years"] else 24)),
                        help=f"Number of {time_range} to generate data for"
                    )
                
                with col2:
                    st.subheader("🏪 Merchant Configuration")
                    num_merchants = st.slider("Number of Merchants", 1, 10, 3)
                    transactions_per_period = st.slider(
                        f"Transactions per {time_range[:-1]}", 
                        10, 
                        2000, 
                        100 if time_range == "hours" else (500 if time_range == "days" else 1000),
                        help="Base number of transactions per time period"
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    seed = st.number_input("Random Seed", value=42, help="For reproducible results")
                with col2:
                    include_downtime = st.checkbox("Include Artificial Downtime", value=True, help="Simulate realistic connector downtime")
                
                if st.button("🚀 Generate Enhanced Data", type="primary"):
                    with st.spinner("Generating enhanced realistic transaction data..."):
                        enhanced_generator = EnhancedRealisticTransactionGenerator(seed=seed)
                        all_transactions = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(num_merchants):
                            merchant_id = f"merchant_{i+1:03d}"
                            status_text.text(f"Generating enhanced data for {merchant_id}...")
                            
                            # Use the time range generator
                            # Map UI values to TimeRange enum values
                            time_range_mapping = {
                                "hours": TimeRange.HOURS,
                                "days": TimeRange.DAYS,
                                "weeks": TimeRange.WEEKS,
                                "months": TimeRange.MONTHS,
                                "years": TimeRange.YEARS
                            }
                            
                            transactions = enhanced_generator.generate_time_range_dataset(
                                merchant_id, 
                                time_range_mapping[time_range], 
                                duration, 
                                transactions_per_period
                            )
                            all_transactions.extend(transactions)
                            
                            progress_bar.progress((i + 1) / num_merchants)
                        
                        # Convert to DataFrame
                        df = pd.DataFrame([t.model_dump() for t in all_transactions])
                        
                        # Save to file
                        os.makedirs('data', exist_ok=True)
                        df.to_csv('data/realistic_all_transactions.csv', index=False)
                        
                        # Generate enhanced report
                        report = enhanced_generator.generate_analysis_report(all_transactions)
                        
                        status_text.text("✅ Enhanced data generation completed!")
                        progress_bar.progress(1.0)
                        
                        # Show enhanced summary
                        st.success("🎉 Enhanced data generated successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", f"{len(all_transactions):,}")
                            st.metric("Success Rate", f"{report['overall_success_rate']:.1%}")
                        with col2:
                            st.metric("Total Volume", f"${report['total_volume']:,.0f}")
                            st.metric("Processing Cost", f"${report['total_cost']:,.2f}")
                        with col3:
                            st.metric("Avg Processing Time", f"{report['avg_processing_time']:.0f}ms")
                            st.metric("Unique Statuses", len(report['status_distribution']))
                        
                        # Show status distribution
                        st.subheader("📊 Enhanced Status Distribution")
                        status_df = pd.DataFrame(list(report['status_distribution'].items()), columns=['Status', 'Count'])
                        status_df = status_df[status_df['Count'] > 0].sort_values('Count', ascending=False)
                        
                        fig = px.bar(status_df, x='Status', y='Count', 
                                   title="Transaction Status Distribution (Enhanced Generator)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.rerun()
            
            else:
                # Basic generator (legacy)
                st.warning("📊 Using basic generator with limited statuses")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_merchants = st.slider("Number of Merchants", 1, 10, 3)
                with col2:
                    transactions_per_merchant = st.slider("Transactions per Merchant", 100, 5000, 1000)
                with col3:
                    seed = st.number_input("Random Seed", value=42, help="For reproducible results")
                
                if st.button("🚀 Generate Basic Data", type="primary"):
                    new_df = generate_batch_data(num_merchants, transactions_per_merchant, seed)
                    st.success("✅ Basic data generated successfully!")
                    st.rerun()
        
        with tab2:
            if df is not None:
                st.subheader("📁 Current Dataset Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Merchants", df['merchant_id'].nunique())
                with col3:
                    st.metric("Connectors", df['connector_id'].nunique())
                with col4:
                    st.metric("Date Range", f"{(pd.to_datetime(df['timestamp']).max() - pd.to_datetime(df['timestamp']).min()).days} days")
                
                # Download options
                st.subheader("💾 Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"transaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"transaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col3:
                    # Summary report
                    summary = {
                        "total_transactions": len(df),
                        "success_rate": (df['status'] == 'success').mean(),
                        "total_volume": df['amount'].sum(),
                        "avg_processing_time": df['processing_time_ms'].mean(),
                        "merchants": df['merchant_id'].nunique(),
                        "connectors": df['connector_id'].nunique()
                    }
                    summary_json = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary_json,
                        file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # Filters
                st.subheader("🔍 Filter Data")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    merchant_filter = st.selectbox("Filter by Merchant", ["All"] + list(df['merchant_id'].unique()))
                
                with col2:
                    status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
                
                with col3:
                    connector_filter = st.selectbox("Filter by Connector", ["All"] + list(df['connector_id'].unique()))
                
                with col4:
                    payment_method_filter = st.selectbox("Filter by Payment Method", ["All"] + list(df['payment_method'].unique()))
                
                # Apply filters
                filtered_df = df.copy()
                
                if merchant_filter != "All":
                    filtered_df = filtered_df[filtered_df['merchant_id'] == merchant_filter]
                
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['status'] == status_filter]
                
                if connector_filter != "All":
                    filtered_df = filtered_df[filtered_df['connector_id'] == connector_filter]
                
                if payment_method_filter != "All":
                    filtered_df = filtered_df[filtered_df['payment_method'] == payment_method_filter]
                
                # Show filter results
                if len(filtered_df) != len(df):
                    st.success(f"🔍 Filtered to {len(filtered_df):,} transactions (from {len(df):,} total)")
                else:
                    st.info(f"📊 Showing all {len(df):,} transactions")
                
                # Data preview with pagination
                st.subheader("👀 Data Preview")
                
                # Pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    rows_per_page = st.selectbox("Rows per page", [50, 100, 250, 500, 1000], index=1)
                
                with col2:
                    total_pages = (len(filtered_df) - 1) // rows_per_page + 1
                    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                
                with col3:
                    st.write(f"Total: {len(filtered_df):,} rows")
                
                # Calculate start and end indices
                start_idx = (page_number - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(filtered_df))
                
                # Display paginated data
                st.dataframe(
                    filtered_df.iloc[start_idx:end_idx],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show pagination info
                st.info(f"📄 Showing rows {start_idx + 1:,} to {end_idx:,} of {len(filtered_df):,} total transactions (Page {page_number} of {total_pages})")
                
            else:
                st.info("No data available. Generate new data to get started.")
        
        with tab3:
            st.subheader("⚙️ Application Settings")
            
            # Clear cache
            if st.button("🗑️ Clear All Data"):
                if st.checkbox("I understand this will delete all data"):
                    st.session_state.real_time_data = []
                    if os.path.exists('data/realistic_all_transactions.csv'):
                        os.remove('data/realistic_all_transactions.csv')
                    st.success("All data cleared!")
                    st.rerun()
            
            # App info
            st.subheader("ℹ️ Application Information")
            st.write(f"**Version:** 2.0.0 Enhanced")
            st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Data Directory:** {os.path.abspath('data')}")

if __name__ == "__main__":
    main()
