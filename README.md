# AI-Based System Sleep Manager

An AI system that monitors SUT/NUC lab machines and automatically puts them to
sleep when they are idle, reducing power consumption.  It uses an **LSTM neural
network** to forecast system states (Idle / Low Load / High Load) and a **fuzzy
logic controller** to prevent catastrophic sleep events.

## Architecture

```
Zabbix Agent ──► Zabbix Server ──► data_pipeline.py ──► PostgreSQL
                                                            │
                                      data_preprocessing.py ◄┘
                                             │
                                        lstm_model.py  (train / predict)
                                             │
                                        fuzzy_logic.py (sleep gate)
                                             │
                                       sleep_manager.py (sleep command)
```

### Components

| Module | Description |
|---|---|
| `config.py` | All configuration: thresholds, DB settings, model hyper-parameters |
| `data_pipeline.py` | Pulls metrics from Zabbix API and stores them in PostgreSQL |
| `data_preprocessing.py` | Labels data (Idle/Low/High), scales features, creates LSTM sequences |
| `lstm_model.py` | Builds, trains and runs inference with a two-layer LSTM classifier |
| `fuzzy_logic.py` | Fuzzy controller that gates the sleep decision using idle probability and consecutive idle count |
| `sleep_manager.py` | Monitors system state in real-time and triggers OS sleep (Windows & Ubuntu) |
| `main.py` | CLI entry-point with sub-commands: `generate`, `train`, `monitor`, `pipeline` |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Simulation Data

```bash
python main.py generate -o data/simulation_data.csv
```

### 3. Train the LSTM Model

```bash
python main.py train --data data/simulation_data.csv --epochs 50
```

### 4. Run the Sleep Monitor

```bash
python main.py monitor
```

### 5. Run the Zabbix → PostgreSQL Pipeline (optional)

```bash
python main.py pipeline
```

## System State Thresholds

| Metric | Idle | Low Load | High Load |
|---|---|---|---|
| CPU Utilization | ≤ 5 % | ≤ 50 % | > 50 % |
| Memory Utilization | ≤ 20 % | ≤ 60 % | > 60 % |
| Disk Read/Write | ≤ 1 KB/s | ≤ 50 KB/s | > 50 KB/s |
| Network In/Out | ≤ 5 KB/s | ≤ 100 KB/s | > 100 KB/s |
| System Idle Time | ≥ 95 % | ≥ 50 % | < 50 % |

## Fuzzy Logic

The fuzzy controller takes two inputs:
- **Idle probability** from the LSTM model output (0–1)
- **Consecutive idle count** — how many consecutive polling intervals the system
  was predicted idle

And outputs a **sleep decision** score (0–1). The system is put to sleep only
when the score exceeds the threshold (default 0.5), which requires both a high
idle probability *and* a sustained idle period.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Supported Operating Systems

- **Windows** — uses `rundll32.exe powrprof.dll,SetSuspendState`
- **Ubuntu / Linux** — uses `systemctl suspend`