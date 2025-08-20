"""
Routing Optimization System - Web UI
A comprehensive web application for generating, viewing, and analyzing realistic transaction data.
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

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generation.realistic_transaction_generator import RealisticTransactionGenerator
from src.models import TransactionStatus

# Configure Streamlit page
st.set_page_config(
    page_title="Routing Optimization System",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .failure-metric {
        border-left-color: #dc3545;
    }
    .cost-metric {
        border-left-color: #ffc107;
    }
    .volume-metric {
        border-left-color: #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

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

def generate_new_data(num_merchants, transactions_per_merchant, seed):
    """Generate new transaction data."""
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

def create_overview_metrics(df):
    """Create overview metrics cards."""
    total_transactions = len(df)
    successful_transactions = len(df[df['status'] == 'success'])
    failed_transactions = len(df[df['status'] == 'failed'])
    success_rate = (successful_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    total_volume = df['amount'].sum()
    total_cost = df[df['status'] == 'success']['cost'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card volume-metric">
            <h3>Total Transactions</h3>
            <h2>{total_transactions:,}</h2>
            <p>Across {df['merchant_id'].nunique()} merchants</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h3>Success Rate</h3>
            <h2>{success_rate:.1f}%</h2>
            <p>{successful_transactions:,} successful transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card volume-metric">
            <h3>Total Volume</h3>
            <h2>${total_volume:,.2f}</h2>
            <p>Across all currencies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card cost-metric">
            <h3>Processing Cost</h3>
            <h2>${total_cost:,.2f}</h2>
            <p>{(total_cost/total_volume)*100:.3f}% of volume</p>
        </div>
        """, unsafe_allow_html=True)

def create_status_distribution_chart(df):
    """Create transaction status distribution chart."""
    status_counts = df['status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Transaction Status Distribution",
        color_discrete_map={
            'success': '#28a745',
            'failed': '#dc3545'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_connector_performance_chart(df):
    """Create connector performance analysis chart."""
    connector_stats = df.groupby('connector_id').agg({
        'id': 'count',
        'status': lambda x: (x == 'success').sum(),
        'amount': 'sum',
        'cost': 'sum'
    }).rename(columns={'id': 'total_transactions', 'status': 'successful_transactions'})
    
    connector_stats['success_rate'] = (connector_stats['successful_transactions'] / connector_stats['total_transactions']) * 100
    connector_stats = connector_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate by Connector', 'Transaction Volume by Connector', 
                       'Total Cost by Connector', 'Transaction Count by Connector'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Success Rate
    fig.add_trace(
        go.Bar(x=connector_stats['connector_id'], y=connector_stats['success_rate'],
               name='Success Rate (%)', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=connector_stats['connector_id'], y=connector_stats['amount'],
               name='Volume ($)', marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Cost
    fig.add_trace(
        go.Bar(x=connector_stats['connector_id'], y=connector_stats['cost'],
               name='Cost ($)', marker_color='lightyellow'),
        row=2, col=1
    )
    
    # Transaction Count
    fig.add_trace(
        go.Bar(x=connector_stats['connector_id'], y=connector_stats['total_transactions'],
               name='Transactions', marker_color='lightcoral'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Connector Performance Analysis")
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_geographic_distribution_chart(df):
    """Create geographic distribution chart."""
    country_stats = df['country_code'].value_counts().head(10)
    
    fig = px.bar(
        x=country_stats.index,
        y=country_stats.values,
        title="Geographic Distribution (Top 10 Countries)",
        labels={'x': 'Country Code', 'y': 'Number of Transactions'},
        color=country_stats.values,
        color_continuous_scale='viridis'
    )
    fig.update_layout(showlegend=False)
    return fig

def create_payment_method_analysis(df):
    """Create payment method analysis chart."""
    payment_stats = df.groupby('payment_method').agg({
        'id': 'count',
        'status': lambda x: (x == 'success').mean() * 100,
        'amount': 'sum'
    }).rename(columns={'id': 'count', 'status': 'success_rate'})
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Transaction Count', 'Success Rate (%)', 'Total Volume ($)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=payment_stats.index, y=payment_stats['count'], name='Count'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=payment_stats.index, y=payment_stats['success_rate'], name='Success Rate'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=payment_stats.index, y=payment_stats['amount'], name='Volume'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Payment Method Analysis")
    return fig

def create_risk_analysis_chart(df):
    """Create risk score analysis chart."""
    # Create risk buckets
    df_copy = df.copy()
    df_copy['risk_bucket'] = pd.cut(df_copy['risk_score'], 
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    risk_stats = df_copy.groupby('risk_bucket').agg({
        'id': 'count',
        'status': lambda x: (x == 'success').mean() * 100
    }).rename(columns={'id': 'count', 'status': 'success_rate'})
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Transaction Count by Risk Level', 'Success Rate by Risk Level'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig.add_trace(
        go.Bar(x=risk_stats.index, y=risk_stats['count'], 
               name='Count', marker_color='lightblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=risk_stats.index, y=risk_stats['success_rate'], 
               name='Success Rate', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Risk Score Analysis")
    return fig

def create_failure_analysis_chart(df):
    """Create failure analysis chart."""
    failed_df = df[df['status'] == 'failed']
    if len(failed_df) == 0:
        return None
    
    failure_reasons = failed_df['failure_reason'].value_counts().head(10)
    
    fig = px.bar(
        x=failure_reasons.values,
        y=failure_reasons.index,
        orientation='h',
        title="Top Failure Reasons",
        labels={'x': 'Number of Failures', 'y': 'Failure Reason'},
        color=failure_reasons.values,
        color_continuous_scale='reds'
    )
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_time_series_chart(df):
    """Create time series analysis chart."""
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    
    hourly_stats = df_copy.groupby('hour').agg({
        'id': 'count',
        'status': lambda x: (x == 'success').mean() * 100
    }).rename(columns={'id': 'count', 'status': 'success_rate'})
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Hourly Transaction Volume', 'Hourly Success Rate'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['count'], 
                  mode='lines+markers', name='Transaction Count'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hourly_stats.index, y=hourly_stats['success_rate'], 
                  mode='lines+markers', name='Success Rate (%)', line_color='red'),
        row=2, col=1
    )
    
    fig.update_layout(height=500, showlegend=False, title_text="Hourly Transaction Patterns")
    fig.update_xaxes(title_text="Hour of Day")
    return fig

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔄 Routing Optimization System</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    st.sidebar.title("🎛️ Controls")
    
    # Data Generation Section
    st.sidebar.header("📊 Data Generation")
    
    num_merchants = st.sidebar.slider("Number of Merchants", 1, 10, 3)
    transactions_per_merchant = st.sidebar.slider("Transactions per Merchant", 100, 2000, 1000)
    seed = st.sidebar.number_input("Random Seed", value=42, help="For reproducible results")
    
    if st.sidebar.button("🚀 Generate New Data", type="primary"):
        df = generate_new_data(num_merchants, transactions_per_merchant, seed)
        st.rerun()
    
    # Load existing data
    df = load_data()
    
    if df is None:
        st.warning("⚠️ No transaction data found. Please generate new data using the sidebar controls.")
        st.info("💡 Click 'Generate New Data' to create realistic transaction data for analysis.")
        return
    
    # Data Overview Section
    st.sidebar.header("📈 Data Overview")
    st.sidebar.metric("Total Transactions", f"{len(df):,}")
    st.sidebar.metric("Date Range", f"{df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
    st.sidebar.metric("Merchants", df['merchant_id'].nunique())
    st.sidebar.metric("Connectors", df['connector_id'].nunique())
    
    # Download Data
    st.sidebar.header("💾 Export Data")
    csv_data = df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"transaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "📈 Performance", "🛠️ Raw Data"])
    
    with tab1:
        st.header("📊 Transaction Overview")
        
        # Overview metrics
        create_overview_metrics(df)
        
        st.markdown("---")
        
        # Charts in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_status_distribution_chart(df), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_geographic_distribution_chart(df), use_container_width=True)
        
        # Time series analysis
        st.plotly_chart(create_time_series_chart(df), use_container_width=True)
    
    with tab2:
        st.header("🔍 Detailed Analysis")
        
        # Payment method analysis
        st.plotly_chart(create_payment_method_analysis(df), use_container_width=True)
        
        # Risk analysis
        st.plotly_chart(create_risk_analysis_chart(df), use_container_width=True)
        
        # Failure analysis
        failure_chart = create_failure_analysis_chart(df)
        if failure_chart:
            st.plotly_chart(failure_chart, use_container_width=True)
        else:
            st.info("No failed transactions to analyze.")
    
    with tab3:
        st.header("📈 Connector Performance")
        
        # Connector performance
        st.plotly_chart(create_connector_performance_chart(df), use_container_width=True)
        
        # Detailed connector stats table
        st.subheader("📋 Connector Statistics")
        
        connector_stats = df.groupby('connector_id').agg({
            'id': 'count',
            'status': lambda x: (x == 'success').sum(),
            'amount': ['sum', 'mean'],
            'cost': 'sum',
            'processing_time_ms': 'mean'
        }).round(2)
        
        connector_stats.columns = ['Total Transactions', 'Successful', 'Total Volume', 'Avg Amount', 'Total Cost', 'Avg Processing Time (ms)']
        connector_stats['Success Rate (%)'] = (connector_stats['Successful'] / connector_stats['Total Transactions'] * 100).round(1)
        connector_stats['Avg Cost per Transaction'] = (connector_stats['Total Cost'] / connector_stats['Successful']).round(4)
        
        st.dataframe(connector_stats, use_container_width=True)
    
    with tab4:
        st.header("🛠️ Raw Transaction Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            merchant_filter = st.selectbox("Filter by Merchant", ["All"] + list(df['merchant_id'].unique()))
        
        with col2:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
        
        with col3:
            connector_filter = st.selectbox("Filter by Connector", ["All"] + list(df['connector_id'].unique()))
        
        # Apply filters
        filtered_df = df.copy()
        
        if merchant_filter != "All":
            filtered_df = filtered_df[filtered_df['merchant_id'] == merchant_filter]
        
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        
        if connector_filter != "All":
            filtered_df = filtered_df[filtered_df['connector_id'] == connector_filter]
        
        st.write(f"Showing {len(filtered_df):,} of {len(df):,} transactions")
        
        # Display data
        st.dataframe(
            filtered_df.head(1000),  # Limit to 1000 rows for performance
            use_container_width=True,
            hide_index=True
        )
        
        if len(filtered_df) > 1000:
            st.info("📝 Showing first 1000 rows. Use filters to narrow down the data or download the full dataset.")

if __name__ == "__main__":
    main()
