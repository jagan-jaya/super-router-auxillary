# Interactive Data Management Tool

## Overview

The Interactive Data Management Tool is an enhanced feature for the Routing Optimization System that allows merchants to create custom transaction datasets by adjusting connector performance patterns through an intuitive drag-and-drop interface.

## Features

### 🎛️ Interactive Connector Configuration
- **Volume Pattern Adjustment**: Modify hourly transaction volume patterns for each connector
- **Success Rate Customization**: Adjust success rates throughout the day
- **Real-time Visualization**: See changes reflected immediately in interactive charts
- **Quick Adjustments**: Use sliders for rapid configuration changes
- **Downtime Simulation**: Simulate connector outages and performance issues

### 📊 Data Generation Options
- **Standard Data Generation**: Use the existing realistic transaction generator
- **Custom Data Generation**: Create data based on your custom connector configurations
- **Flexible Merchant Configuration**: Set custom merchant IDs and transaction volumes

### 📈 Advanced Analytics
- **Performance Comparison**: Compare connector performance side-by-side
- **Hourly Pattern Analysis**: Visualize transaction patterns throughout the day
- **Real-time Metrics**: Get instant feedback on generated data quality

## How to Use

### 1. Launch the Application

```bash
# Run the standalone interactive data management tool
streamlit run interactive_data_management_app.py

# Or run the enhanced integrated version
streamlit run enhanced_interactive_app.py
```

### 2. Configure Connectors

1. **Navigate to the "Configure Connectors" tab**
2. **Enable/Disable Connectors**: Toggle which connectors you want to include
3. **Adjust Volume Patterns**: 
   - Use the volume multiplier to scale overall transaction volume
   - Apply peak hour boosts for business hour traffic spikes
   - Manually input 24-hour patterns if needed
4. **Modify Success Rates**:
   - Adjust overall success rates up or down
   - Simulate downtime periods with reduced success rates
   - Fine-tune hourly variations

### 3. Generate Custom Data

1. **Set Merchant Configuration**:
   - Enter a custom merchant ID
   - Choose the number of transactions to generate
2. **Review Configuration Summary**: Verify your connector settings
3. **Generate Data**: Click the generate button to create custom transaction data

### 4. Analyze Results

1. **View Data Preview**: See summary metrics and performance comparisons
2. **Examine Hourly Patterns**: Analyze how your configurations affected transaction distribution
3. **Download Data**: Export the generated dataset for further analysis

## Configuration Options

### Volume Pattern Configuration

- **Volume Multiplier**: Scale base transaction volume (0.1x to 3.0x)
- **Peak Hours Boost**: Increase volume during peak business hours
- **Manual Input**: Enter exact hourly values for precise control

### Success Rate Configuration

- **Success Rate Adjustment**: Modify base success rates (-10% to +10%)
- **Downtime Simulation**: Add realistic outage periods (0-8 hours)
- **Hourly Variations**: Fine-tune success rates for each hour

### Realistic Patterns

The system generates realistic patterns based on:
- **Business Hours**: Higher volume during 9 AM - 6 PM
- **Geographic Patterns**: Different patterns for different regions
- **Connector Characteristics**: Each connector has unique baseline performance
- **Random Variations**: Natural fluctuations in real-world data

## Use Cases

### 1. Testing Routing Algorithms
Create specific scenarios to test how routing algorithms perform under different conditions:
- High-volume periods with varying success rates
- Connector outages and failover scenarios
- Geographic distribution testing

### 2. Performance Benchmarking
Generate datasets that match your expected traffic patterns:
- Seasonal volume variations
- Regional performance differences
- Payment method preferences

### 3. Stress Testing
Simulate extreme conditions:
- Multiple connector outages
- Sudden volume spikes
- Extended downtime periods

### 4. Training Data Generation
Create diverse datasets for machine learning models:
- Balanced success/failure distributions
- Various transaction patterns
- Different merchant categories

## Technical Details

### Data Generation Process

1. **Configuration Loading**: System loads connector configurations from session state
2. **Pattern Application**: Hourly patterns are applied to base connector characteristics
3. **Transaction Generation**: Individual transactions are created with realistic attributes
4. **Success Rate Application**: Custom success rates are applied based on time of day
5. **Data Validation**: Generated data is validated for consistency and realism

### Connector Models

Each connector includes:
- **Base Success Rate**: Default performance level
- **Cost Structure**: Transaction processing costs
- **Volume Capacity**: Maximum transaction handling capability
- **Geographic Coverage**: Supported regions
- **Payment Methods**: Supported payment types

### Pattern Generation

Volume and success rate patterns are generated using:
- **Realistic Multipliers**: Based on actual payment processing patterns
- **Random Variations**: Natural fluctuations using normal distributions
- **Business Logic**: Peak hours, geographic preferences, etc.

## File Structure

```
routing-optimization-system/
├── interactive_data_management_app.py    # Standalone interactive tool
├── enhanced_interactive_app.py           # Integrated version with existing features
├── src/
│   ├── data_generation/
│   │   └── realistic_transaction_generator.py
│   └── models.py
└── data/                                 # Generated datasets
```

## Dependencies

- **Streamlit**: Web application framework
- **Plotly**: Interactive charting
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Faker**: Realistic data generation

## Best Practices

### 1. Configuration Management
- Start with default patterns and make incremental adjustments
- Test configurations with small datasets before generating large volumes
- Save successful configurations for reuse

### 2. Data Quality
- Ensure success rates remain realistic (70-98%)
- Maintain reasonable volume distributions
- Validate generated data against expected patterns

### 3. Performance Optimization
- Generate data in batches for large datasets
- Use appropriate random seeds for reproducible results
- Monitor system resources during generation

## Troubleshooting

### Common Issues

1. **No Connectors Enabled**: Ensure at least one connector is enabled before generating data
2. **Invalid Patterns**: Check that hourly patterns contain exactly 24 values
3. **Memory Issues**: Reduce transaction count for large datasets

### Performance Tips

- Use smaller datasets for testing configurations
- Enable only necessary connectors to improve performance
- Clear session state if experiencing issues

## Future Enhancements

- **Drag-and-Drop Chart Editing**: Direct manipulation of chart lines
- **Template Management**: Save and load configuration templates
- **Batch Processing**: Generate multiple datasets with different configurations
- **Advanced Analytics**: More sophisticated pattern analysis tools
- **Export Formats**: Support for additional data formats (JSON, Parquet, etc.)

## Support

For questions or issues with the Interactive Data Management Tool, please refer to the main project documentation or create an issue in the project repository.
