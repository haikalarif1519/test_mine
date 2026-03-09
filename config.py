"""
Configuration for the AI-based System Sleep Manager.

Contains thresholds for system state classification, database settings,
Zabbix settings, LSTM model parameters, and fuzzy logic parameters.
"""

# ---------------------------------------------------------------------------
# Zabbix Configuration
# ---------------------------------------------------------------------------
ZABBIX_URL = "http://localhost/zabbix/api_jsonrpc.php"
ZABBIX_USER = "Admin"
ZABBIX_PASSWORD = "zabbix"

# Zabbix item keys for monitored metrics
ZABBIX_ITEM_KEYS = {
    "cpu_utilization": "system.cpu.util[,idle]",
    "memory_utilization": "vm.memory.utilization",
    "disk_read": "vfs.dev.read.rate[sda]",
    "disk_write": "vfs.dev.write.rate[sda]",
    "network_received": "net.if.in[eth0]",
    "network_sent": "net.if.out[eth0]",
    "system_idle_time": "system.cpu.util[,idle]",
}

# ---------------------------------------------------------------------------
# PostgreSQL Configuration
# ---------------------------------------------------------------------------
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "system_monitor"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# ---------------------------------------------------------------------------
# System State Thresholds
# ---------------------------------------------------------------------------
# Thresholds used to label each data sample as Idle, Low Load, or High Load.
# A sample is labelled based on the *highest* matching category across all
# metrics.  Values are expressed as percentages (0-100) for utilization
# metrics, bytes/sec for disk and network metrics.

THRESHOLDS = {
    "cpu_utilization": {
        "idle": 5.0,       # <= 5 %
        "low_load": 50.0,  # <= 50 %
        # > 50 % → high_load
    },
    "memory_utilization": {
        "idle": 20.0,      # <= 20 %
        "low_load": 60.0,  # <= 60 %
    },
    "disk_read": {
        "idle": 1_000.0,        # <= 1 KB/s
        "low_load": 50_000.0,   # <= 50 KB/s
    },
    "disk_write": {
        "idle": 1_000.0,
        "low_load": 50_000.0,
    },
    "network_received": {
        "idle": 5_000.0,        # <= 5 KB/s
        "low_load": 100_000.0,  # <= 100 KB/s
    },
    "network_sent": {
        "idle": 5_000.0,
        "low_load": 100_000.0,
    },
    "system_idle_time": {
        "idle": 95.0,       # >= 95 % idle → system is idle
        "low_load": 50.0,   # >= 50 % idle → low load
    },
}

# ---------------------------------------------------------------------------
# State Labels
# ---------------------------------------------------------------------------
STATE_LABELS = {"idle": 0, "low_load": 1, "high_load": 2}
STATE_NAMES = {v: k for k, v in STATE_LABELS.items()}

# ---------------------------------------------------------------------------
# LSTM Model Parameters
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 10          # Number of time-steps fed to the LSTM
LSTM_UNITS = 64               # Hidden units in each LSTM layer
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NUM_FEATURES = 7              # Number of input features
NUM_CLASSES = 3               # idle, low_load, high_load

# ---------------------------------------------------------------------------
# Fuzzy Logic Parameters
# ---------------------------------------------------------------------------
# Idle probability range from LSTM output (0 – 1)
FUZZY_IDLE_PROB_LOW = 0.0
FUZZY_IDLE_PROB_HIGH = 1.0

# Consecutive idle count range (number of consecutive idle predictions)
FUZZY_CONSEC_LOW = 0
FUZZY_CONSEC_HIGH = 10

# Sleep decision output range (0 – 1, where > 0.5 → sleep)
FUZZY_SLEEP_LOW = 0.0
FUZZY_SLEEP_HIGH = 1.0
FUZZY_SLEEP_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Sleep Manager
# ---------------------------------------------------------------------------
IDLE_TIMEOUT_MINUTES = 10     # Put system to sleep after N minutes idle
POLL_INTERVAL_SECONDS = 60    # How often to check system state
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
PIPELINE_POLL_INTERVAL_SECONDS = 60   # Pull from Zabbix every 60 s
HISTORY_HOURS = 24                     # How many hours of history to pull
