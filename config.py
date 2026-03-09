"""Configuration for the AI Idle Detection and Sleep Management System."""

import os

# Zabbix API Configuration
ZABBIX_URL = os.environ.get("ZABBIX_URL", "http://your-zabbix-server/api_jsonrpc.php")
ZABBIX_USER = os.environ.get("ZABBIX_USER", "Admin")
ZABBIX_PASSWORD = os.environ.get("ZABBIX_PASSWORD", "zabbix")

# Zabbix item keys for the metrics we collect
ZABBIX_ITEM_KEYS = {
    "cpu_utilization": "system.cpu.util[,idle]",
    "memory_utilization": "vm.memory.size[pused]",
    "disk_read": "vfs.dev.read.rate[sda]",
    "disk_write": "vfs.dev.write.rate[sda]",
    "network_received": "net.if.in[eth0]",
    "network_sent": "net.if.out[eth0]",
    "system_idle_time": "system.cpu.util[,idle]",
}

# Idle detection thresholds
IDLE_THRESHOLD_MINUTES = 10  # Put system to sleep after 10 minutes of idle
MONITORING_INTERVAL_SECONDS = 60  # Check every 60 seconds
IDLE_PROBABILITY_THRESHOLD = 0.7  # Model confidence threshold for idle classification

# History fetch settings
HISTORY_PERIOD_SECONDS = 3600  # Fetch 1 hour of history for training data
HISTORY_LIMIT = 1000  # Max number of history records per item

# Model settings
MODEL_PATH = os.environ.get("MODEL_PATH", "models/idle_detection_model.joblib")
TRAINING_DATA_PATH = os.environ.get("TRAINING_DATA_PATH", "data/training_data.csv")

# Sleep command (executed via Zabbix agent on the remote system)
SLEEP_COMMAND = "systemctl suspend"

# Feature columns used by the model
FEATURE_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "disk_read",
    "disk_write",
    "network_received",
    "network_sent",
    "system_idle_time",
]
