"""
Interactive Data Management Tool for Routing Optimization System
Allows merchants to create custom transaction data by adjusting volume and success rate charts.
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
    page_title="Interactive Data Management - Routing Optimization",
    page_icon="🎛️",
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
</style>
""", unsafe_allow_html=True)

class InteractiveDataManager:
    """Manages interactive data generation with customizable connector performance."""
    
    def __init__(self):
        self.generator = RealisticTransactionGenerator(seed=42)
        self.connectors = self.generator.connectors
        self.time_periods = 24  # 24 hours
        
        # Initialize session state for connector configurations
        if 'connector_configs' not in st.session_state:
            st.session_state.connector_configs = self._initialize_connector_configs()
        
        if 'generated_data' not in st.session_state:
            st.session_state.generated_data = None
    
    def _initialize_connector_configs(self):
        """Initialize default connector configurations."""
        configs = {}
        for connector in self.connectors:
            # Generate realistic hourly patterns
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
            # Volume typically peaks during business hours
            multipliers = [
                0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8,  # 0-7 AM
                1.0, 1.2, 1.1, 0.9, 0.8, 0.9, 1.1, 1.2,  # 8-15 (8AM-3PM)
                1.0, 0.9, 0.8, 1.1, 1.2, 1.0, 0.7, 0.5   # 16-23 (4PM-11PM)
            ]
            return [int(base_value * m) for m in multipliers]
        else:  # success_rate
            # Success rate might vary slightly throughout the day
            base_rate = min(base_value, 95)  # Cap at 95%
            variation = np.random.normal(0, 2, 24)  # Small random variations
            pattern = [max(70, min(98, base_rate + v)) for v in variation]
            return pattern
    
    def render_connector_configuration(self):
        """Render the interactive connector configuration interface."""
        st.header("🎛️ Connector Performance Configuration")
        st.write("Adjust volume and success rate patterns for each connector by dragging the chart lines.")
        
        # Connector selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Connectors")
            for connector_id, config in st.session_state.connector_configs.items():
                with st.container():
                    st.markdown(f"""
                    <div class="connector-card">
                        <h4>{config['name']}</h4>
                        <p>Base Success Rate: {config['base_success_rate']:.1%}</p>
                        <p>Cost: ${config['base_cost']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enable/disable toggle
                    config['is_enabled'] = st.checkbox(
                        f"Enable {config['name']}", 
                        value=config['is_enabled'],
                        key=f"enable_{connector_id}"
                    )
        
        with col2:
            st.subheader("Performance Charts")
            
            # Create tabs for different connectors
            connector_tabs = st.tabs([config['name'] for config in st.session_state.connector_configs.values()])
            
            for i, (connector_id, config) in enumerate(st.session_state.connector_configs.items()):
                with connector_tabs[i]:
                    if config['is_enabled']:
                        self._render_connector_charts(connector_id, config)
                    else:
                        st.info(f"{config['name']} is disabled. Enable it to configure performance patterns.")
    
    def _render_connector_charts(self, connector_id, config):
        """Render interactive charts for a specific connector."""
        hours = list(range(24))
        
        # Volume Chart
        st.subheader(f"📊 {config['name']} - Hourly Volume")
        
        # Create volume adjustment sliders
        col1, col2 = st.columns([3, 1])
        
        with col1:
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Scatter(
                x=hours,
                y=config['volume_pattern'],
                mode='lines+markers',
                name='Transaction Volume',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ))
            
            volume_fig.update_layout(
                title=f"Hourly Transaction Volume - {config['name']}",
                xaxis_title="Hour of Day",
                yaxis_title="Transactions per Hour",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(volume_fig, use_container_width=True, key=f"volume_{connector_id}")
        
        with col2:
            st.write("**Quick Adjustments**")
            
            # Volume multiplier
            volume_multiplier = st.slider(
                "Volume Multiplier",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key=f"vol_mult_{connector_id}"
            )
            
            if st.button(f"Apply Volume Multiplier", key=f"apply_vol_{connector_id}"):
                base_pattern = config['volume_pattern'].copy()
                config['volume_pattern'] = [int(v * volume_multiplier) for v in base_pattern]
                st.rerun()
            
            # Peak hour boost
            peak_boost = st.slider(
                "Peak Hours Boost",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                key=f"peak_boost_{connector_id}"
            )
            
            if st.button(f"Apply Peak Boost", key=f"apply_peak_{connector_id}"):
                # Boost hours 9-11 and 14-16 and 19-21
                peak_hours = [9, 10, 11, 14, 15, 16, 19, 20, 21]
                for hour in peak_hours:
                    config['volume_pattern'][hour] = int(config['volume_pattern'][hour] * (1 + peak_boost))
                st.rerun()
        
        # Success Rate Chart
        st.subheader(f"✅ {config['name']} - Success Rate")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            success_fig = go.Figure()
            success_fig.add_trace(go.Scatter(
                x=hours,
                y=config['success_rate_pattern'],
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='#28a745', width=3),
                marker=dict(size=8)
            ))
            
            success_fig.update_layout(
                title=f"Hourly Success Rate - {config['name']}",
                xaxis_title="Hour of Day",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[70, 100]),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(success_fig, use_container_width=True, key=f"success_{connector_id}")
        
        with col2:
            st.write("**Success Rate Adjustments**")
            
            # Success rate adjustment
            success_adjustment = st.slider(
                "Success Rate Adjustment",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key=f"success_adj_{connector_id}"
            )
            
            if st.button(f"Apply Success Adjustment", key=f"apply_success_{connector_id}"):
                config['success_rate_pattern'] = [
                    max(70, min(98, rate + success_adjustment)) 
                    for rate in config['success_rate_pattern']
                ]
                st.rerun()
            
            # Add downtime simulation
            if st.button(f"Simulate Downtime", key=f"downtime_{connector_id}"):
                # Random 2-4 hour downtime
                start_hour = np.random.randint(0, 20)
                duration = np.random.randint(2, 5)
                for hour in range(start_hour, min(start_hour + duration, 24)):
                    config['success_rate_pattern'][hour] = np.random.randint(20, 40)
                st.rerun()
        
        # Manual value input
        with st.expander(f"Manual Value Input - {config['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Volume Pattern (24 hours)**")
                volume_input = st.text_area(
                    "Enter 24 comma-separated values:",
                    value=",".join(map(str, config['volume_pattern'])),
                    key=f"volume_input_{connector_id}"
                )
                
                if st.button(f"Update Volume", key=f"update_vol_{connector_id}"):
                    try:
                        values = [int(v.strip()) for v in volume_input.split(',')]
                        if len(values) == 24:
                            config['volume_pattern'] = values
                            st.success("Volume pattern updated!")
                            st.rerun()
                        else:
                            st.error("Please enter exactly 24 values.")
                    except ValueError:
                        st.error("Please enter valid integer values.")
            
            with col2:
                st.write("**Success Rate Pattern (24 hours)**")
                success_input = st.text_area(
                    "Enter 24 comma-separated values (70-98):",
                    value=",".join(map(str, config['success_rate_pattern'])),
                    key=f"success_input_{connector_id}"
                )
                
                if st.button(f"Update Success Rate", key=f"update_success_{connector_id}"):
                    try:
                        values = [float(v.strip()) for v in success_input.split(',')]
                        if len(values) == 24 and all(70 <= v <= 98 for v in values):
                            config['success_rate_pattern'] = values
                            st.success("Success rate pattern updated!")
                            st.rerun()
                        else:
                            st.error("Please enter exactly 24 values between 70 and 98.")
                    except ValueError:
                        st.error("Please enter valid numeric values.")
    
    def generate_custom_data(self, merchant_id, num_transactions):
        """Generate transaction data based on custom connector configurations."""
        st.info("🔄 Generating custom transaction data based on your configurations...")
        
        # Create a custom generator with modified connector settings
        custom_generator = RealisticTransactionGenerator(seed=42)
        
        # Update connector success rates and volumes based on configurations
        for connector in custom_generator.connectors:
            if connector.id in st.session_state.connector_configs:
                config = st.session_state.connector_configs[connector.id]
                if config['is_enabled']:
                    # Use average success rate from pattern
                    avg_success_rate = np.mean(config['success_rate_pattern']) / 100
                    connector.success_rate = avg_success_rate
                    
                    # Update volume capacity based on pattern
                    avg_volume = np.mean(config['volume_pattern'])
                    connector.volume_capacity = int(avg_volume * 24)  # Daily capacity
        
        # Generate transactions with time-based patterns
        transactions = []
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_transactions):
            # Distribute transactions across 24 hours based on volume patterns
            hour_weights = []
            for connector_id, config in st.session_state.connector_configs.items():
                if config['is_enabled']:
                    hour_weights.extend(config['volume_pattern'])
            
            if not hour_weights:
                hour_weights = [1] * 24  # Fallback if no connectors enabled
            
            # Select random hour based on volume distribution
            hour = np.random.choice(24, p=np.array(hour_weights[:24]) / np.sum(hour_weights[:24]))
            
            # Create timestamp for selected hour
            transaction_time = start_time + timedelta(hours=hour, minutes=np.random.randint(0, 60))
            
            # Generate transaction
            transaction = custom_generator.generate_realistic_transaction(merchant_id, transaction_time)
            
            # Apply custom success rate for the hour
            if transaction.connector_id in st.session_state.connector_configs:
                config = st.session_state.connector_configs[transaction.connector_id]
                if config['is_enabled']:
                    custom_success_rate = config['success_rate_pattern'][hour] / 100
                    
                    # Re-determine transaction status based on custom success rate
                    if np.random.random() < custom_success_rate:
                        transaction.status = TransactionStatus.SUCCESS
                        transaction.cost = transaction.amount * config['base_cost']
                        transaction.failure_reason = None
                    else:
                        transaction.status = TransactionStatus.FAILED
                        transaction.cost = 0.0
                        transaction.failure_reason = np.random.choice([
                            "insufficient_funds", "card_declined", "network_timeout",
                            "fraud_suspected", "issuer_unavailable"
                        ])
            
            transactions.append(transaction)
            
            # Update progress
            progress_bar.progress((i + 1) / num_transactions)
            if i % 100 == 0:
                status_text.text(f"Generated {i + 1}/{num_transactions} transactions...")
        
        status_text.text("✅ Transaction generation completed!")
        progress_bar.progress(1.0)
        
        return transactions
    
    def render_data_preview(self, transactions):
        """Render preview of generated data."""
        if not transactions:
            return
        
        df = pd.DataFrame([t.model_dump() for t in transactions])
        
        st.subheader("📊 Generated Data Preview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Transactions", len(df))
        
        with col2:
            success_rate = (df['status'] == 'success').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            total_volume = df['amount'].sum()
            st.metric("Total Volume", f"${total_volume:,.2f}")
        
        with col4:
            total_cost = df[df['status'] == 'success']['cost'].sum()
            st.metric("Processing Cost", f"${total_cost:.2f}")
        
        # Connector performance comparison
        st.subheader("🔍 Connector Performance Comparison")
        
        connector_stats = df.groupby('connector_id').agg({
            'id': 'count',
            'status': lambda x: (x == 'success').mean() * 100,
            'amount': 'sum',
            'cost': 'sum'
        }).round(2)
        
        connector_stats.columns = ['Transactions', 'Success Rate (%)', 'Volume ($)', 'Cost ($)']
        st.dataframe(connector_stats, use_container_width=True)
        
        # Hourly patterns visualization
        st.subheader("📈 Hourly Transaction Patterns")
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_stats = df.groupby(['hour', 'connector_id']).agg({
            'id': 'count',
            'status': lambda x: (x == 'success').mean() * 100
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            volume_fig = px.line(
                hourly_stats, 
                x='hour', 
                y='id', 
                color='connector_id',
                title="Hourly Transaction Volume by Connector",
                labels={'id': 'Transaction Count', 'hour': 'Hour of Day'}
            )
            st.plotly_chart(volume_fig, use_container_width=True)
        
        with col2:
            success_fig = px.line(
                hourly_stats, 
                x='hour', 
                y='status', 
                color='connector_id',
                title="Hourly Success Rate by Connector",
                labels={'status': 'Success Rate (%)', 'hour': 'Hour of Day'}
            )
            st.plotly_chart(success_fig, use_container_width=True)
        
        # Data download
        st.subheader("💾 Download Generated Data")
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"custom_transaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🎛️ Interactive Data Management Tool</h1>', unsafe_allow_html=True)
    
    # Initialize the data manager
    data_manager = InteractiveDataManager()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Data Generation Controls")
    
    merchant_id = st.sidebar.text_input("Merchant ID", value="merchant_custom_001")
    num_transactions = st.sidebar.slider("Number of Transactions", 100, 5000, 1000)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎛️ Configure Connectors", "🚀 Generate Data", "📊 Data Preview"])
    
    with tab1:
        data_manager.render_connector_configuration()
        
        # Reset configurations
        if st.button("🔄 Reset All Configurations"):
            st.session_state.connector_configs = data_manager._initialize_connector_configs()
            st.success("All configurations reset to defaults!")
            st.rerun()
    
    with tab2:
        st.header("🚀 Generate Custom Transaction Data")
        
        st.write("Generate transaction data based on your custom connector configurations.")
        
        # Configuration summary
        st.subheader("📋 Current Configuration Summary")
        
        enabled_connectors = [
            config['name'] for config in st.session_state.connector_configs.values() 
            if config['is_enabled']
        ]
        
        if enabled_connectors:
            st.success(f"Enabled Connectors: {', '.join(enabled_connectors)}")
            
            # Show configuration preview
            config_preview = {}
            for connector_id, config in st.session_state.connector_configs.items():
                if config['is_enabled']:
                    config_preview[config['name']] = {
                        'Avg Daily Volume': sum(config['volume_pattern']),
                        'Avg Success Rate': f"{np.mean(config['success_rate_pattern']):.1f}%",
                        'Peak Volume Hour': config['volume_pattern'].index(max(config['volume_pattern'])),
                        'Min Success Rate': f"{min(config['success_rate_pattern']):.1f}%"
                    }
            
            st.json(config_preview)
        else:
            st.warning("No connectors are enabled. Please enable at least one connector in the configuration tab.")
        
        # Generate data button
        if st.button("🚀 Generate Custom Data", type="primary", disabled=not enabled_connectors):
            transactions = data_manager.generate_custom_data(merchant_id, num_transactions)
            st.session_state.generated_data = transactions
            st.success(f"Generated {len(transactions)} transactions successfully!")
    
    with tab3:
        st.header("📊 Generated Data Preview")
        
        if st.session_state.generated_data:
            data_manager.render_data_preview(st.session_state.generated_data)
        else:
            st.info("No data generated yet. Please generate data in the 'Generate Data' tab.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Interactive Data Management Tool - Routing Optimization System</p>
        <p>Drag and adjust connector performance patterns to create custom transaction datasets</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
