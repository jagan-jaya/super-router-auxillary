"""
Enhanced Routing Optimization System with Interactive Data Management
Combines the existing analytics with new interactive data generation capabilities.
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
from src.models import TransactionStatus, Connector, ConnectorType, PaymentMethod, Currency

# Configure Streamlit page
st.set_page_config(
    page_title="Enhanced Routing Optimization System",
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
    .connector-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
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
    .chart-container {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    .data-management-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDataManager:
    """Enhanced data manager with interactive capabilities."""
    
    def __init__(self):
        self.generator = RealisticTransactionGenerator(seed=42)
        self.connectors = self.generator.connectors
        
        # Initialize session state
        if 'connector_configs' not in st.session_state:
            st.session_state.connector_configs = self._initialize_connector_configs()
        
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
        
        if 'data_source' not in st.session_state:
            st.session_state.data_source = 'existing'  # 'existing' or 'custom'
    
    def _initialize_connector_configs(self):
        """Initialize default connector configurations."""
        configs = {}
        for connector in self.connectors:
            base_volume = np.random.randint(50, 200)
            volume_pattern = self._generate_realistic_pattern(base_volume, 'volume')
            success_rate_pattern = self._generate_realistic_pattern(connector.success_rate * 100, 'success_rate')
            
            configs[connector.id] = {
                'name': connector.name,
                'base_success_rate': connector.success_rate,
                'base_cost': connector.cost_per_transaction,
                'volume_pattern': volume_pattern,
                'success_rate_pattern': success_rate_pattern,
                'is_enabled': True
            }
        
        return configs
    
    def _generate_realistic_pattern(self, base_value, pattern_type):
        """Generate realistic hourly patterns for volume or success rate."""
        if pattern_type == 'volume':
            multipliers = [
                0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8,  # 0-7 AM
                1.0, 1.2, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2,  # 8-15 (8AM-3PM)
                1.0, 0.9, 0.8, 1.1, 1.2, 1.0, 0.7, 0.5   # 16-23 (4PM-11PM)
            ]
            return [int(base_value * m) for m in multipliers]
        else:  # success_rate
            base_rate = min(base_value, 95)
            variation = np.random.normal(0, 2, 24)
            pattern = [max(70, min(98, base_rate + v)) for v in variation]
            return pattern
    
    def load_existing_data(self):
        """Load existing transaction data if available."""
        try:
            if os.path.exists('data/realistic_all_transactions.csv'):
                return pd.read_csv('data/realistic_all_transactions.csv')
            else:
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    
    def generate_new_data(self, num_merchants, transactions_per_merchant, seed):
        """Generate new transaction data using existing generator."""
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
    
    def render_interactive_data_management(self):
        """Render the interactive data management interface."""
        st.markdown("""
        <div class="data-management-header">
            <h2>🎛️ Interactive Data Management</h2>
            <p>Create custom transaction datasets by adjusting connector performance patterns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data source selection
        data_source = st.radio(
            "Choose Data Source:",
            ["Use Existing Data", "Create Custom Data"],
            index=0 if st.session_state.data_source == 'existing' else 1,
            horizontal=True
        )
        
        st.session_state.data_source = 'existing' if data_source == "Use Existing Data" else 'custom'
        
        if st.session_state.data_source == 'custom':
            self._render_custom_data_interface()
        else:
            self._render_existing_data_interface()
    
    def _render_existing_data_interface(self):
        """Render interface for existing data generation."""
        st.subheader("📊 Standard Data Generation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_merchants = st.slider("Number of Merchants", 1, 10, 3)
        
        with col2:
            transactions_per_merchant = st.slider("Transactions per Merchant", 100, 2000, 1000)
        
        with col3:
            seed = st.number_input("Random Seed", value=42)
        
        if st.button("🚀 Generate Standard Data", type="primary"):
            df = self.generate_new_data(num_merchants, transactions_per_merchant, seed)
            st.session_state.generated_data = df
            st.success(f"Generated {len(df)} transactions successfully!")
    
    def _render_custom_data_interface(self):
        """Render interface for custom data generation."""
        st.subheader("🎛️ Custom Data Generation")
        
        # Merchant configuration
        col1, col2 = st.columns(2)
        
        with col1:
            merchant_id = st.text_input("Merchant ID", value="merchant_custom_001")
        
        with col2:
            num_transactions = st.slider("Number of Transactions", 100, 5000, 1000)
        
        # Connector configuration tabs
        st.subheader("Connector Performance Configuration")
        
        connector_tabs = st.tabs([config['name'] for config in st.session_state.connector_configs.values()])
        
        for i, (connector_id, config) in enumerate(st.session_state.connector_configs.items()):
            with connector_tabs[i]:
                self._render_connector_config_compact(connector_id, config)
        
        # Generate custom data
        st.subheader("🚀 Generate Custom Data")
        
        enabled_connectors = [
            config['name'] for config in st.session_state.connector_configs.values() 
            if config['is_enabled']
        ]
        
        if enabled_connectors:
            st.success(f"Enabled Connectors: {', '.join(enabled_connectors)}")
            
            if st.button("🚀 Generate Custom Data", type="primary"):
                transactions = self._generate_custom_data(merchant_id, num_transactions)
                df = pd.DataFrame([t.model_dump() for t in transactions])
                st.session_state.generated_data = df
                st.success(f"Generated {len(transactions)} custom transactions successfully!")
        else:
            st.warning("Please enable at least one connector.")
    
    def _render_connector_config_compact(self, connector_id, config):
        """Render compact connector configuration interface."""
        # Enable/disable toggle
        config['is_enabled'] = st.checkbox(
            f"Enable {config['name']}", 
            value=config['is_enabled'],
            key=f"enable_{connector_id}"
        )
        
        if not config['is_enabled']:
            st.info(f"{config['name']} is disabled.")
            return
        
        # Quick configuration options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Volume Configuration**")
            
            volume_multiplier = st.slider(
                "Volume Multiplier",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key=f"vol_mult_{connector_id}"
            )
            
            peak_boost = st.slider(
                "Peak Hours Boost",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key=f"peak_boost_{connector_id}"
            )
            
            if st.button(f"Apply Volume Changes", key=f"apply_vol_{connector_id}"):
                base_pattern = self._generate_realistic_pattern(100, 'volume')
                config['volume_pattern'] = [int(v * volume_multiplier) for v in base_pattern]
                
                # Apply peak boost
                peak_hours = [9, 10, 11, 14, 15, 16, 19, 20, 21]
                for hour in peak_hours:
                    config['volume_pattern'][hour] = int(config['volume_pattern'][hour] * (1 + peak_boost))
                
                st.success("Volume pattern updated!")
        
        with col2:
            st.write("**Success Rate Configuration**")
            
            success_adjustment = st.slider(
                "Success Rate Adjustment",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key=f"success_adj_{connector_id}"
            )
            
            downtime_hours = st.slider(
                "Downtime Duration (hours)",
                min_value=0,
                max_value=8,
                value=0,
                key=f"downtime_hours_{connector_id}"
            )
            
            if st.button(f"Apply Success Rate Changes", key=f"apply_success_{connector_id}"):
                base_pattern = self._generate_realistic_pattern(config['base_success_rate'] * 100, 'success_rate')
                config['success_rate_pattern'] = [
                    max(70, min(98, rate + success_adjustment)) 
                    for rate in base_pattern
                ]
                
                # Apply downtime
                if downtime_hours > 0:
                    start_hour = np.random.randint(0, 24 - downtime_hours)
                    for hour in range(start_hour, start_hour + downtime_hours):
                        config['success_rate_pattern'][hour] = np.random.randint(20, 40)
                
                st.success("Success rate pattern updated!")
        
        # Show current patterns
        with st.expander(f"View Current Patterns - {config['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=config['volume_pattern'],
                    mode='lines+markers',
                    name='Volume',
                    line=dict(color='#1f77b4')
                ))
                volume_fig.update_layout(
                    title="Hourly Volume Pattern",
                    xaxis_title="Hour",
                    yaxis_title="Transactions",
                    height=250
                )
                st.plotly_chart(volume_fig, use_container_width=True)
            
            with col2:
                success_fig = go.Figure()
                success_fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=config['success_rate_pattern'],
                    mode='lines+markers',
                    name='Success Rate',
                    line=dict(color='#28a745')
                ))
                success_fig.update_layout(
                    title="Hourly Success Rate Pattern",
                    xaxis_title="Hour",
                    yaxis_title="Success Rate (%)",
                    height=250
                )
                st.plotly_chart(success_fig, use_container_width=True)
    
    def _generate_custom_data(self, merchant_id, num_transactions):
        """Generate custom transaction data based on configurations."""
        custom_generator = RealisticTransactionGenerator(seed=42)
        
        # Update connector settings
        for connector in custom_generator.connectors:
            if connector.id in st.session_state.connector_configs:
                config = st.session_state.connector_configs[connector.id]
                if config['is_enabled']:
                    avg_success_rate = np.mean(config['success_rate_pattern']) / 100
                    connector.success_rate = avg_success_rate
                    avg_volume = np.mean(config['volume_pattern'])
                    connector.volume_capacity = int(avg_volume * 24)
        
        # Generate transactions
        transactions = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        progress_bar = st.progress(0)
        
        for i in range(num_transactions):
            # Select hour based on volume patterns
            hour_weights = []
            for connector_id, config in st.session_state.connector_configs.items():
                if config['is_enabled']:
                    hour_weights.extend(config['volume_pattern'])
            
            if hour_weights:
                hour = np.random.choice(24, p=np.array(hour_weights[:24]) / np.sum(hour_weights[:24]))
            else:
                hour = np.random.randint(0, 24)
            
            transaction_time = start_time + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            transaction = custom_generator.generate_realistic_transaction(merchant_id, transaction_time)
            
            # Apply custom success rate
            if transaction.connector_id in st.session_state.connector_configs:
                config = st.session_state.connector_configs[transaction.connector_id]
                if config['is_enabled']:
                    custom_success_rate = config['success_rate_pattern'][hour] / 100
                    
                    if np.random.random() < custom_success_rate:
                        transaction.status = TransactionStatus.SUCCESS
                        transaction.cost = transaction.amount * config['base_cost']
                        transaction.failure_reason = None
                    else:
                        transaction.status = TransactionStatus.FAILED
                        transaction.cost = 0.0
                        transaction.failure_reason = np.random.choice([
                            "insufficient_funds", "card_declined", "network_timeout"
                        ])
            
            transactions.append(transaction)
            progress_bar.progress((i + 1) / num_transactions)
        
        return transactions

# Import existing chart functions from the original app
def create_overview_metrics(df):
    """Create overview metrics cards."""
    total_transactions = len(df)
    successful_transactions = len(df[df['status'] == 'success'])
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

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🔄 Enhanced Routing Optimization System</h1>', unsafe_allow_html=True)
    
    # Initialize the enhanced data manager
    data_manager = EnhancedDataManager()
    
    # Sidebar for controls
    st.sidebar.title("🎛️ System Controls")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎛️ Data Management", 
        "📊 Overview", 
        "📈 Performance", 
        "🛠️ Raw Data"
    ])
    
    with tab1:
        data_manager.render_interactive_data_management()
    
    # Get current data
    if st.session_state.generated_data is not None:
        df = st.session_state.generated_data
    else:
        df = data_manager.load_existing_data()
    
    if df is None:
        st.warning("⚠️ No transaction data found. Please generate data in the Data Management tab.")
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
    
    with tab2:
        st.header("📊 Transaction Overview")
        
        # Overview metrics
        create_overview_metrics(df)
        
        st.markdown("---")
        
        # Status distribution
        status_counts = df['status'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Geographic distribution
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
            st.plotly_chart(fig, use_container_width=True)
    
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
            filtered_df.head(1000),
            use_container_width=True,
            hide_index=True
        )
        
        if len(filtered_df) > 1000:
            st.info("📝 Showing first 1000 rows. Use filters to narrow down the data or download the full dataset.")

if __name__ == "__main__":
    main()
