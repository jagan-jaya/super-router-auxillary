# Routing Optimization System

A comprehensive payment routing optimization system that maximizes output by combining different routing algorithms including Success Rate (SR), Volume Distribution (V), Least Cost Routing (LCR), and Rule-based routing.

## Features

- **Real-time Transaction Data Generation**: Simulates realistic payment transaction scenarios
- **Multiple Routing Algorithms**: 
  - Success Rate (SR) optimization
  - Volume Distribution (V) management
  - Least Cost Routing (LCR)
  - Rule-based routing
- **Algorithm Combinations**: SR+V, LCR+V, LCR+V+SR, Rule+LCR, Rule+SR, Rule+LCR+SR
- **Performance Analytics**: Comprehensive analysis and visualization of routing performance
- **Real-time Optimization**: Dynamic routing decisions based on live transaction data

## Goals

1. **Maximize Auth Uplift**: Improve authorization success rates
2. **Reduce Transaction Costs**: Optimize cost through intelligent routing
3. **Adhere to Merchant Rules**: Respect merchant-configured routing rules
4. **Maintain Volume Distribution**: Ensure proper volume split across connectors

## Project Structure

```
routing-optimization-system/
├── src/
│   ├── data_generation/          # Transaction data generation
│   ├── routing_algorithms/       # Core routing algorithms
│   ├── optimization/            # Optimization strategies
│   ├── analytics/               # Performance analysis
│   └── api/                     # REST API endpoints
├── tests/                       # Unit and integration tests
├── config/                      # Configuration files
├── data/                        # Generated data storage
└── notebooks/                   # Jupyter notebooks for analysis
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample transaction data:
```bash
python -m src.data_generation.transaction_generator
```

3. Run optimization analysis:
```bash
python -m src.optimization.routing_optimizer
```

4. Start the API server:
```bash
python3 run_app.py  
```

## Routing Algorithms

### Success Rate (SR)
Optimizes routing based on historical success rates of connectors for specific transaction types.

### Volume Distribution (V)
Manages volume split across connectors (e.g., Stripe 80%, Adyen 20%) while maintaining performance.

### Least Cost Routing (LCR)
Selects the most cost-effective routing path, preferring local networks over global networks.

### Rule-based Routing
Applies merchant-configured rules for specific transaction criteria.

## Algorithm Combinations

The system supports various combinations to maximize different objectives:

- **SR + V**: Balance success rate with volume distribution
- **LCR + V**: Minimize costs while maintaining volume targets
- **LCR + V + SR**: Triple optimization for cost, volume, and success
- **Rule + LCR**: Apply rules with cost optimization
- **Rule + SR**: Apply rules with success rate optimization
- **Rule + LCR + SR**: Complete optimization with rule compliance

## License

MIT License
