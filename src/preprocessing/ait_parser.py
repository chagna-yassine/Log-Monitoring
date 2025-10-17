"""
AIT-LDS Log Parser

Parses AIT Log Data Sets with multi-host structure and attack labels.
Handles various log types: Apache, auth, DNS, VPN, audit, Suricata, syslog, etc.
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import yaml


class AITLogParser:
    """
    Parses AIT-LDS logs from multiple hosts and log types.
    
    The AIT-LDS dataset has the following structure:
    - gather/{host}/logs/{log_type}/... - Raw log files
    - labels/{host}/logs/{log_type}/... - JSON label files
    - dataset.yml - Dataset metadata
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ait_config = config['ait_dataset']
        self.output_config = config['output']['ait']
        self.drain_config = config['preprocessing']['drain']
        
        # Setup paths
        self.base_path = Path(self.ait_config['base_path'])
        self.dataset_name = self.ait_config['selected_dataset']
        self.dataset_path = self.base_path / self.dataset_name
        self.output_path = Path(self.output_config['base_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset metadata
        self.dataset_yml = self.dataset_path / 'dataset.yml'
        self.dataset_info = self._load_dataset_info()
        
        # Log type patterns
        self.log_patterns = {
            'apache_access': r'^(\d+\.\d+\.\d+\.\d+) - - \[([^\]]+)\] "([^"]+)" (\d+) (\d+) "([^"]*)" "([^"]*)"$',
            'apache_error': r'^\[([^\]]+)\] \[([^\]]+)\] \[client ([^\]]+)\] ([^:]+): (.+)$',
            'auth': r'^(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\w+)(\[[\d]+\])?: (.+)$',
            'dns': r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (.+)$',
            'audit': r'^type=(\w+) msg=audit\(([^)]+)\): (.+)$',
            'syslog': r'^(\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (\S+) (\S+): (.+)$',
            'suricata': r'^(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\.\d+) \[([^\]]+)\] ([^:]+): (.+)$'
        }
    
    def _load_dataset_info(self) -> Dict[str, Any]:
        """Load dataset metadata from dataset.yml"""
        if not self.dataset_yml.exists():
            print(f"Warning: dataset.yml not found at {self.dataset_yml}")
            return {}
        
        try:
            with open(self.dataset_yml, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load dataset.yml: {e}")
            return {}
    
    def parse_log_file(self, log_file: Path, log_type: str) -> pd.DataFrame:
        """
        Parse a single log file based on its type.
        
        Args:
            log_file: Path to the log file
            log_type: Type of log (apache_access, auth, etc.)
            
        Returns:
            DataFrame with parsed log entries
        """
        entries = []
        
        if not log_file.exists():
            return pd.DataFrame()
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse based on log type
                    parsed_entry = self._parse_log_line(line, log_type, line_num)
                    if parsed_entry:
                        parsed_entry['LineNumber'] = line_num
                        parsed_entry['LogFile'] = str(log_file)
                        parsed_entry['LogType'] = log_type
                        entries.append(parsed_entry)
        
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            return pd.DataFrame()
        
        return pd.DataFrame(entries)
    
    def _parse_log_line(self, line: str, log_type: str, line_num: int) -> Dict[str, Any]:
        """Parse a single log line based on its type."""
        entry = {
            'Timestamp': None,
            'Source': None,
            'Content': line,
            'Parsed': False
        }
        
        if log_type in self.log_patterns:
            pattern = self.log_patterns[log_type]
            match = re.match(pattern, line)
            
            if match:
                entry['Parsed'] = True
                
                if log_type == 'apache_access':
                    entry.update({
                        'Timestamp': match.group(2),
                        'Source': match.group(1),
                        'Method': match.group(3).split()[0] if ' ' in match.group(3) else match.group(3),
                        'Status': match.group(4),
                        'Size': match.group(5),
                        'UserAgent': match.group(7)
                    })
                
                elif log_type == 'apache_error':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Level': match.group(2),
                        'Source': match.group(3),
                        'ErrorType': match.group(4),
                        'Message': match.group(5)
                    })
                
                elif log_type == 'auth':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Source': match.group(2),
                        'Process': match.group(3),
                        'Message': match.group(5)
                    })
                
                elif log_type == 'audit':
                    entry.update({
                        'EventType': match.group(1),
                        'Timestamp': match.group(2),
                        'Details': match.group(3)
                    })
                
                elif log_type == 'syslog':
                    entry.update({
                        'Timestamp': match.group(1),
                        'Source': match.group(2),
                        'Process': match.group(3),
                        'Message': match.group(4)
                    })
        
        return entry
    
    def load_attack_labels(self, label_file: Path) -> Dict[int, List[str]]:
        """
        Load attack labels from JSON file.
        
        Args:
            label_file: Path to the label file
            
        Returns:
            Dictionary mapping line numbers to attack labels
        """
        labels = {}
        
        if not label_file.exists():
            return labels
        
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        label_data = json.loads(line)
                        line_num = label_data.get('line')
                        attack_labels = label_data.get('labels', [])
                        
                        if line_num is not None:
                            labels[line_num] = attack_labels
                    
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            print(f"Error loading labels from {label_file}: {e}")
        
        return labels
    
    def parse_all_logs(self) -> pd.DataFrame:
        """
        Parse all logs from all hosts in the dataset.
        
        Returns:
            Combined DataFrame with all parsed logs
        """
        print(f"Parsing AIT-LDS logs from: {self.dataset_path}")
        print(f"Dataset: {self.dataset_name}")
        
        all_logs = []
        
        # Get all hosts from gather directory
        gather_dir = self.dataset_path / 'gather'
        if not gather_dir.exists():
            print(f"Error: gather directory not found: {gather_dir}")
            return pd.DataFrame()
        
        hosts = [d for d in gather_dir.iterdir() if d.is_dir()]
        print(f"Found {len(hosts)} hosts: {[h.name for h in hosts]}")
        
        total_files = 0
        parsed_files = 0
        
        for host_dir in hosts:
            host_name = host_dir.name
            logs_dir = host_dir / 'logs'
            
            if not logs_dir.exists():
                continue
            
            print(f"\nProcessing host: {host_name}")
            
            # Get all log files
            log_files = []
            for log_type_dir in logs_dir.iterdir():
                if log_type_dir.is_dir():
                    log_type = log_type_dir.name
                    for log_file in log_type_dir.rglob('*'):
                        if log_file.is_file():
                            log_files.append((log_file, log_type))
            
            print(f"  Found {len(log_files)} log files")
            
            for log_file, log_type in log_files:
                total_files += 1
                
                # Parse log file
                log_df = self.parse_log_file(log_file, log_type)
                
                if not log_df.empty:
                    # Add host information
                    log_df['Host'] = host_name
                    
                    # Load corresponding labels
                    label_file = self._get_label_file(log_file)
                    if label_file.exists():
                        labels = self.load_attack_labels(label_file)
                        
                        # Add labels to log entries
                        log_df['AttackLabels'] = log_df['LineNumber'].map(labels)
                        log_df['IsAttack'] = log_df['AttackLabels'].notna()
                        
                        # Convert attack labels to binary (1 if any attack, 0 if normal)
                        log_df['Label'] = log_df['IsAttack'].astype(int)
                    else:
                        # No labels found - assume normal
                        log_df['AttackLabels'] = None
                        log_df['IsAttack'] = False
                        log_df['Label'] = 0
                    
                    all_logs.append(log_df)
                    parsed_files += 1
                    
                    if parsed_files % 10 == 0:
                        print(f"  Parsed {parsed_files}/{total_files} files...")
        
        if not all_logs:
            print("No logs were successfully parsed!")
            return pd.DataFrame()
        
        # Combine all logs
        combined_df = pd.concat(all_logs, ignore_index=True)
        
        print(f"\nParsing complete:")
        print(f"  Total files processed: {total_files}")
        print(f"  Successfully parsed: {parsed_files}")
        print(f"  Total log entries: {len(combined_df):,}")
        
        # Statistics
        if 'Label' in combined_df.columns:
            attack_count = combined_df['Label'].sum()
            normal_count = len(combined_df) - attack_count
            attack_rate = (attack_count / len(combined_df)) * 100
            
            print(f"  Normal entries: {normal_count:,} ({100-attack_rate:.2f}%)")
            print(f"  Attack entries: {attack_count:,} ({attack_rate:.2f}%)")
        
        # Show log type distribution
        if 'LogType' in combined_df.columns:
            log_type_counts = combined_df['LogType'].value_counts()
            print(f"\nLog type distribution:")
            for log_type, count in log_type_counts.items():
                print(f"  {log_type}: {count:,}")
        
        return combined_df
    
    def _get_label_file(self, log_file: Path) -> Path:
        """Get the corresponding label file for a log file."""
        # Convert gather path to labels path
        relative_path = log_file.relative_to(self.dataset_path / 'gather')
        label_file = self.dataset_path / 'labels' / relative_path
        return label_file
    
    def save_structured_logs(self, df: pd.DataFrame) -> Path:
        """Save parsed logs to structured CSV file."""
        output_file = self.output_path / self.output_config['structured_log']
        
        print(f"Saving structured logs to: {output_file}")
        df.to_csv(output_file, index=False)
        
        return output_file
    
    def extract_params(self, log_content: str) -> List[str]:
        """
        Extract parameters from log content using regex patterns.
        This is used by Drain parser for template extraction.
        """
        # Common parameter patterns for various log types
        patterns = [
            r'\b\d+\.\d+\.\d+\.\d+\b',  # IP addresses
            r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b',  # Timestamps
            r'\b\d+\b',  # Numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IP addresses (alternative)
            r'\b[a-fA-F0-9]{32}\b',  # MD5 hashes
            r'\b[a-fA-F0-9]{40}\b',  # SHA1 hashes
        ]
        
        content = log_content
        for pattern in patterns:
            content = re.sub(pattern, '<*>', content)
        
        return [content]
