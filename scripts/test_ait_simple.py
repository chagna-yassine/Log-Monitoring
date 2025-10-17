"""
Simple AIT-LDS Test

Creates a small synthetic AIT-LDS dataset for testing when the full dataset
download is not available or fails.
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import json
from datetime import datetime, timedelta


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_synthetic_ait_dataset(output_dir: Path, num_logs: int = 1000):
    """Create a synthetic AIT-LDS dataset for testing."""
    print(f"Creating synthetic AIT-LDS dataset with {num_logs} log entries...")
    
    # Create directory structure
    dataset_dir = output_dir / "russellmitchell"
    gather_dir = dataset_dir / "gather"
    labels_dir = dataset_dir / "labels"
    
    # Create host directories
    hosts = ["web_server", "mail_server", "intranet_server", "firewall"]
    log_types = ["apache_access", "apache_error", "auth", "audit", "syslog"]
    
    for host in hosts:
        host_dir = gather_dir / host / "logs"
        host_labels_dir = labels_dir / host / "logs"
        
        for log_type in log_types:
            log_dir = host_dir / log_type
            label_dir = host_labels_dir / log_type
            log_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic logs
    logs_per_host = num_logs // len(hosts)
    attack_ratio = 0.1  # 10% attack logs
    
    all_logs = []
    all_labels = []
    
    for host_idx, host in enumerate(hosts):
        print(f"  Generating logs for {host}...")
        
        # Generate normal logs
        normal_logs = []
        for i in range(int(logs_per_host * (1 - attack_ratio))):
            timestamp = datetime.now() - timedelta(hours=i//10)
            
            # Create different log types
            log_type = log_types[i % len(log_types)]
            
            if log_type == "apache_access":
                log_entry = f"192.168.1.100 - - [{timestamp.strftime('%d/%b/%Y:%H:%M:%S')} +0000] \"GET /index.html HTTP/1.1\" 200 1234 \"-\" \"Mozilla/5.0\""
            elif log_type == "apache_error":
                log_entry = f"[{timestamp.strftime('%a %b %d %H:%M:%S %Y')}] [error] [client 192.168.1.100] File does not exist: /var/www/html/missing.html"
            elif log_type == "auth":
                log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} {host} sshd[12345]: Accepted password for user from 192.168.1.100 port 22 ssh2"
            elif log_type == "audit":
                log_entry = f"type=USER_AUTH msg=audit({timestamp.timestamp():.3f}:12345): pid=12345 uid=0 auid=4294967295 ses=4294967295 msg='op=PAM:authentication acct=\"root\" exe=\"/bin/su\" hostname=? addr=? terminal=/dev/pts/1 res=success'"
            else:  # syslog
                log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} {host} systemd[1]: Started User Manager for UID 1000."
            
            normal_logs.append({
                'host': host,
                'log_type': log_type,
                'content': log_entry,
                'timestamp': timestamp.isoformat(),
                'line_number': len(normal_logs) + 1,
                'is_attack': False
            })
        
        # Generate attack logs
        attack_logs = []
        attack_types = ["scan", "webshell_upload", "password_cracking", "privilege_escalation"]
        
        for i in range(int(logs_per_host * attack_ratio)):
            timestamp = datetime.now() - timedelta(hours=i//5)
            attack_type = attack_types[i % len(attack_types)]
            log_type = log_types[i % len(log_types)]
            
            if attack_type == "scan":
                log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} {host} suricata[12345]: [1:2000001:1] ET SCAN Nmap NULL scan [Classification: Attempted Information Leak] [Priority: 2] {{TCP}} 192.168.1.200:12345 -> 192.168.1.100:80"
            elif attack_type == "webshell_upload":
                log_entry = f"192.168.1.200 - - [{timestamp.strftime('%d/%b/%Y:%H:%M:%S')} +0000] \"POST /wp-content/uploads/shell.php HTTP/1.1\" 200 1234 \"-\" \"Mozilla/5.0\""
            elif attack_type == "password_cracking":
                log_entry = f"{timestamp.strftime('%b %d %H:%M:%S')} {host} sshd[12345]: Failed password for root from 192.168.1.200 port 22 ssh2"
            else:  # privilege_escalation
                log_entry = f"type=USER_AUTH msg=audit({timestamp.timestamp():.3f}:12345): pid=12345 uid=0 auid=4294967295 ses=4294967295 msg='op=PAM:authentication acct=\"root\" exe=\"/bin/su\" hostname=? addr=? terminal=/dev/pts/1 res=success'"
            
            attack_logs.append({
                'host': host,
                'log_type': log_type,
                'content': log_entry,
                'timestamp': timestamp.isoformat(),
                'line_number': len(normal_logs) + len(attack_logs) + 1,
                'is_attack': True,
                'attack_type': attack_type
            })
        
        # Combine logs and save to files
        host_logs = normal_logs + attack_logs
        all_logs.extend(host_logs)
        
        # Save logs by type
        for log_type in log_types:
            type_logs = [log for log in host_logs if log['log_type'] == log_type]
            if type_logs:
                log_file = gather_dir / host / "logs" / log_type / f"{log_type}.log"
                with open(log_file, 'w') as f:
                    for log in type_logs:
                        f.write(log['content'] + '\n')
                
                # Create corresponding label file
                attack_logs_type = [log for log in type_logs if log['is_attack']]
                if attack_logs_type:
                    label_file = labels_dir / host / "logs" / log_type / f"{log_type}.log"
                    with open(label_file, 'w') as f:
                        for log in attack_logs_type:
                            label_entry = {
                                "line": log['line_number'],
                                "labels": [log['attack_type']],
                                "rules": {log['attack_type']: [f"attacker.{log['attack_type']}.{log_type}.rule"]}
                            }
                            f.write(json.dumps(label_entry) + '\n')
    
    # Create dataset.yml
    dataset_yml = dataset_dir / "dataset.yml"
    dataset_info = {
        'simulation_start': '2022-01-21 00:00:00',
        'simulation_end': '2022-01-25 00:00:00',
        'attack_start': '2022-01-24 03:01:00',
        'attack_end': '2022-01-24 04:39:00',
        'description': 'Synthetic AIT-LDS dataset for testing'
    }
    
    with open(dataset_yml, 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False)
    
    print(f"✓ Created synthetic AIT-LDS dataset:")
    print(f"  Location: {dataset_dir}")
    print(f"  Total logs: {len(all_logs)}")
    print(f"  Attack logs: {sum(1 for log in all_logs if log['is_attack'])}")
    print(f"  Hosts: {hosts}")
    print(f"  Log types: {log_types}")
    
    return dataset_dir


def main():
    """Create synthetic AIT-LDS dataset for testing."""
    print("="*70)
    print("SYNTHETIC AIT-LDS DATASET CREATION")
    print("="*70)
    
    # Load configuration
    config = load_config()
    ait_config = config['ait_dataset']
    base_path = Path(ait_config['base_path'])
    
    print(f"\nCreating synthetic dataset at: {base_path}")
    
    # Create synthetic dataset
    dataset_dir = create_synthetic_ait_dataset(base_path, num_logs=1000)
    
    print(f"\n" + "="*70)
    print("SYNTHETIC DATASET CREATION COMPLETE!")
    print("="*70)
    
    print(f"\nYou can now test the AIT-LDS pipeline with this synthetic data:")
    print(f"  1. python scripts/preprocess_ait.py")
    print(f"  2. python scripts/benchmark_ait.py")
    
    print(f"\nNote: This is a small synthetic dataset for testing.")
    print(f"For full evaluation, download the real AIT-LDS dataset when available.")


if __name__ == "__main__":
    main()
