# 🚀 Enhanced Routing Optimization System UI

A comprehensive web application for generating, viewing, and analyzing realistic transaction data in real-time.

## 🌟 Features

### 📊 Dashboard
- **Real-time Metrics**: Live transaction counts, success rates, volume, and processing times
- **Interactive Charts**: Status distribution, geographic analysis, connector performance
- **Enhanced Visualizations**: Gradient-styled metric cards with real-time updates
- **Connector Heatmaps**: Performance analysis by payment method and connector

### ⚡ Real-time Monitor
- **Live Data Generation**: Stream transactions in real-time (1-60 transactions/minute)
- **Real-time Charts**: Live transaction flow visualization
- **Auto-refresh**: Automatic updates every 2 seconds
- **Live Metrics**: Instant success rates, volume, and processing time tracking

### 🔍 Advanced Analytics
- **Time-based Analysis**: Hourly and daily transaction patterns
- **Risk Score Distribution**: Statistical analysis of transaction risk
- **Amount vs Processing Time**: Correlation analysis with scatter plots
- **Statistical Summaries**: Detailed descriptive statistics

### 🤖 ML Insights
- **Predictive Modeling**: Random Forest classifier for transaction success prediction
- **Feature Importance**: Analysis of factors affecting transaction success
- **Model Performance**: Accuracy metrics and feature rankings
- **Automated Encoding**: Smart categorical variable handling

### 🛠️ Data Management
- **Batch Data Generation**: Create large datasets with customizable parameters
- **Multiple Export Formats**: CSV, JSON, and summary reports
- **Data Preview**: Interactive data exploration with filtering
- **Settings Management**: Clear data, view app information

## 🚀 Quick Start

### Option 1: Using the Launcher Script (Recommended)

```bash
# Install dependencies and run enhanced version
python run_app.py --install

# Run enhanced version (default)
python run_app.py

# Run basic version
python run_app.py --mode basic

# Just install dependencies
python run_app.py --install
```

### Option 2: Direct Streamlit Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run enhanced version
streamlit run enhanced_app.py

# Run basic version
streamlit run app.py
```

## 📋 Requirements

### Python Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
fastapi
uvicorn
pydantic
python-dateutil
faker
pytest
streamlit>=1.28.0
plotly>=5.15.0
streamlit-autorefresh
streamlit-aggrid
streamlit-option-menu
watchdog
```

### System Requirements
- Python 3.8+
- 4GB+ RAM (for large datasets)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🎯 Usage Guide

### 1. First Time Setup

1. **Generate Initial Data**:
   - Navigate to "🛠️ Data Management" → "📊 Generate Data"
   - Set number of merchants (1-10)
   - Set transactions per merchant (100-5000)
   - Click "🚀 Generate New Data"

2. **Explore Dashboard**:
   - Go to "📊 Dashboard" to see overview metrics
   - View interactive charts and heatmaps
   - Analyze transaction patterns

### 2. Real-time Monitoring

1. **Start Real-time Generation**:
   - Navigate to "⚡ Real-time Monitor"
   - Set transactions per minute (1-60)
   - Click "🚀 Start Real-time"

2. **Monitor Live Data**:
   - Watch real-time metrics update
   - View live transaction flow charts
   - See recent transactions table

3. **Stop When Done**:
   - Click "⏹️ Stop Real-time" to end generation

### 3. Advanced Analysis

1. **Time-based Patterns**:
   - Go to "🔍 Advanced Analytics"
   - Analyze hourly and daily transaction volumes
   - View risk score distributions

2. **ML Insights**:
   - Navigate to "🤖 ML Insights"
   - View feature importance for transaction success
   - Check model accuracy and predictions

### 4. Data Export

1. **Export Options**:
   - Go to "🛠️ Data Management" → "📁 Current Data"
   - Download as CSV, JSON, or summary report
   - Include timestamps in filenames

## 🎨 UI Features

### Enhanced Styling
- **Gradient Backgrounds**: Beautiful gradient metric cards
- **Real-time Indicators**: Pulsing indicators for live data
- **Responsive Design**: Works on desktop and tablet
- **Dark/Light Themes**: Automatic theme detection

### Interactive Elements
- **Hover Effects**: Rich tooltips and hover information
- **Clickable Charts**: Interactive Plotly visualizations
- **Filtering**: Real-time data filtering and search
- **Auto-refresh**: Seamless real-time updates

## 📊 Data Schema

### Transaction Fields
```python
{
    "id": "unique_transaction_id",
    "merchant_id": "merchant_001",
    "connector_id": "stripe_us",
    "amount": 150.00,
    "currency": "USD",
    "payment_method": "card",
    "country_code": "US",
    "status": "success|failed",
    "timestamp": "2024-01-01T12:00:00Z",
    "processing_time_ms": 250,
    "risk_score": 0.15,
    "cost": 4.50,
    "failure_reason": "insufficient_funds"
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom data directory
export DATA_DIR="/path/to/data"

# Optional: Set custom port for Streamlit
export STREAMLIT_PORT=8501
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

2. **Port Already in Use**:
   ```bash
   # Use different port
   streamlit run enhanced_app.py --server.port 8502
   ```

3. **Memory Issues with Large Datasets**:
   - Reduce number of merchants or transactions
   - Clear data regularly in settings
   - Use filtering in raw data view

4. **Real-time Performance**:
   - Lower transactions per minute
   - Clear real-time data buffer
   - Close other browser tabs

### Performance Tips

1. **For Large Datasets**:
   - Use sampling for scatter plots
   - Limit real-time buffer to 1000 transactions
   - Export data regularly

2. **For Real-time Monitoring**:
   - Start with 10 transactions/minute
   - Monitor system resources
   - Use single browser tab

## 🔄 Version Comparison

| Feature | Basic Version | Enhanced Version |
|---------|---------------|------------------|
| Data Generation | ✅ Batch only | ✅ Batch + Real-time |
| Analytics | ✅ Basic charts | ✅ Advanced + ML |
| UI/UX | ✅ Standard | ✅ Enhanced styling |
| Export | ✅ CSV only | ✅ Multiple formats |
| Real-time | ❌ | ✅ Live monitoring |
| ML Insights | ❌ | ✅ Predictive models |

## 📈 Performance Metrics

### Recommended Limits
- **Batch Generation**: Up to 50,000 transactions
- **Real-time Rate**: 1-60 transactions/minute
- **Real-time Buffer**: 1,000 transactions max
- **Chart Data Points**: 10,000 max for performance

### System Requirements by Dataset Size
- **Small (< 10K transactions)**: 2GB RAM
- **Medium (10K-50K transactions)**: 4GB RAM
- **Large (50K+ transactions)**: 8GB+ RAM

## 🤝 Contributing

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add new visualization functions
4. Update navigation menu
5. Test with sample data
6. Submit pull request

### Custom Analytics
```python
def create_custom_chart(df):
    """Add your custom analytics here."""
    # Your analysis code
    fig = px.your_chart_type(df, ...)
    return fig

# Add to main() function in selected section
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **Pandas**: For data manipulation
- **Scikit-learn**: For machine learning capabilities

---

**Happy Analyzing! 🚀📊**

For support or questions, please open an issue in the repository.
