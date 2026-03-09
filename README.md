# AI-Based Idle Detection and Sleep Management System

An AI-powered system that monitors lab SUT/NUC machines via Zabbix and automatically puts idle systems to sleep to reduce power consumption.

## Overview

Lab systems (SUT/NUC) are always powered on for remote access, leading to high energy consumption. This system uses a **Random Forest classifier** trained on system metrics to detect idle machines and automatically suspends them after **10 minutes** of continuous inactivity.

### Monitored Metrics

- **CPU Utilization** — percentage of CPU in use
- **Memory Utilization** — percentage of RAM in use
- **Disk Read/Write** — disk I/O rates
- **Network Received/Sent** — network I/O rates
- **System Idle Time** — CPU idle percentage

All metrics are collected via the **Zabbix API** (Zabbix agent must be installed on each system).

## Project Structure

```
├── config.py                 # Configuration (Zabbix connection, thresholds)
├── zabbix_client.py          # Zabbix API client for data collection
├── data_preprocessing.py     # Feature engineering and data labeling
├── idle_detection_model.py   # Random Forest idle detection model
├── train_model.py            # Model training script
├── main.py                   # Main monitoring and sleep management loop
├── requirements.txt          # Python dependencies
└── tests/                    # Unit tests
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Zabbix Connection

Set environment variables or edit `config.py`:

```bash
export ZABBIX_URL="http://your-zabbix-server/api_jsonrpc.php"
export ZABBIX_USER="Admin"
export ZABBIX_PASSWORD="zabbix"
```

### 3. Train the Model

Train with synthetic data (for initial testing):

```bash
python train_model.py --synthetic --samples 1000
```

Train with real Zabbix data:

```bash
python train_model.py --zabbix --group "LAB Systems"
```

### 4. Run the Monitor

Start monitoring with dry-run mode (no actual sleep commands):

```bash
python main.py --dry-run
```

Start monitoring for real:

```bash
python main.py --group "LAB Systems"
```

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `IDLE_THRESHOLD_MINUTES` | 10 | Minutes of idle before sleep |
| `MONITORING_INTERVAL_SECONDS` | 60 | Seconds between checks |
| `IDLE_PROBABILITY_THRESHOLD` | 0.7 | Model confidence threshold |

## Running Tests

```bash
python -m pytest tests/ -v
```