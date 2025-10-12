"""
HDFS Log Parser

Parses raw HDFS logs using Drain algorithm and extracts structured information.
"""

import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parsers.drain import Drain


class HDFSLogParser:
    """
    Parser for HDFS log files.
    
    Extracts structured information from raw HDFS logs:
    - Date, Time
    - Process ID (Pid)
    - Log Level
    - Component
    - Content
    - Event ID and Template (via Drain)
    """
    
    # HDFS log format: <Date> <Time> <Pid> <Level> <Component>: <Content>
    LOG_PATTERN = re.compile(
        r'(?P<Date>\d{6})\s+'
        r'(?P<Time>\d{6})\s+'
        r'(?P<Pid>\d+)\s+'
        r'(?P<Level>\w+)\s+'
        r'(?P<Component>[\w\.]+):\s+'
        r'(?P<Content>.*)'
    )
    
    def __init__(
        self,
        depth: int = 4,
        st: float = 0.4,
        rex: Optional[List[str]] = None
    ):
        """
        Initialize HDFS log parser.
        
        Args:
            depth: Drain parser depth
            st: Similarity threshold
            rex: Regular expressions for masking variables
        """
        # Default regex patterns for HDFS logs
        if rex is None:
            rex = [
                r"(?<=blk_)[-\d]+",  # Block IDs
                r"\d+\.\d+\.\d+\.\d+",  # IP addresses
                r"(/[-\w]+)+"  # File paths
            ]
        
        self.drain = Drain(depth=depth, st=st, rex=rex)
        
    def parse_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single log line.
        
        Args:
            line: Raw log line
            
        Returns:
            Dictionary with parsed fields or None if parsing fails
        """
        match = self.LOG_PATTERN.match(line.strip())
        
        if not match:
            return None
        
        return match.groupdict()
    
    def parse_file(
        self,
        log_file: str,
        output_structured: str,
        output_templates: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse an entire HDFS log file.
        
        Args:
            log_file: Path to raw HDFS.log file
            output_structured: Path to save structured log CSV
            output_templates: Path to save templates CSV
            
        Returns:
            Tuple of (structured_df, templates_df)
        """
        print(f"Parsing log file: {log_file}")
        
        parsed_logs = []
        log_id = 0
        failed_lines = 0
        
        # Read and parse log file
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                # Parse log line structure
                parsed = self.parse_line(line)
                
                if parsed is None:
                    failed_lines += 1
                    continue
                
                # Extract event template using Drain
                content = parsed['Content']
                event_id, event_template = self.drain.parse(log_id, content)
                
                # Store structured log entry
                parsed_logs.append({
                    'LineId': log_id,
                    'Date': parsed['Date'],
                    'Time': parsed['Time'],
                    'Pid': parsed['Pid'],
                    'Level': parsed['Level'],
                    'Component': parsed['Component'],
                    'Content': content,
                    'EventId': event_id,
                    'EventTemplate': event_template
                })
                
                log_id += 1
                
                # Progress update
                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines, {log_id:,} valid logs, {failed_lines:,} failed")
        
        print(f"\nParsing complete:")
        print(f"  Total lines processed: {line_num:,}")
        print(f"  Valid logs: {log_id:,}")
        print(f"  Failed lines: {failed_lines:,}")
        print(f"  Unique event templates: {len(self.drain.clusters):,}")
        
        # Create structured log DataFrame
        structured_df = pd.DataFrame(parsed_logs)
        
        # Create templates DataFrame
        templates = self.drain.get_templates()
        cluster_stats = self.drain.get_cluster_stats()
        
        templates_df = pd.DataFrame([
            {
                'EventId': event_id,
                'EventTemplate': template,
                'Occurrences': cluster_stats[event_id]
            }
            for event_id, template in templates.items()
        ])
        
        # Sort by occurrences (most common first)
        templates_df = templates_df.sort_values('Occurrences', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        print(f"\nSaving structured log to: {output_structured}")
        structured_df.to_csv(output_structured, index=False)
        
        print(f"Saving templates to: {output_templates}")
        templates_df.to_csv(output_templates, index=False)
        
        return structured_df, templates_df
    
    def get_statistics(self) -> Dict:
        """
        Get parsing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_clusters': len(self.drain.clusters),
            'cluster_stats': self.drain.get_cluster_stats()
        }

