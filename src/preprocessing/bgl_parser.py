"""
BGL Log Parser

Parses raw BGL logs using Drain algorithm and extracts structured information.
BGL logs have a different format than HDFS logs.
"""

import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from parsers.drain import Drain


class BGLLogParser:
    """
    Parser for BGL (Blue Gene/L) log files.
    
    BGL log format typically:
    - <Timestamp> <NodeID> <Level> <Component>: <Content>
    - Or variations with different field orders
    
    Extracts structured information:
    - Timestamp
    - NodeID
    - Log Level
    - Component
    - Content
    - Event ID and Template (via Drain)
    """
    
    # BGL log format patterns (multiple possible formats)
    BGL_PATTERNS = [
        # Pattern 1: <Timestamp> <NodeID> <Level> <Component>: <Content>
        re.compile(
            r'(?P<Timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<NodeID>\w+)\s+'
            r'(?P<Level>\w+)\s+'
            r'(?P<Component>[\w\.]+):\s+'
            r'(?P<Content>.*)'
        ),
        # Pattern 2: <Timestamp> <Level> <Component> <NodeID>: <Content>
        re.compile(
            r'(?P<Timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<Level>\w+)\s+'
            r'(?P<Component>[\w\.]+)\s+'
            r'(?P<NodeID>\w+):\s+'
            r'(?P<Content>.*)'
        ),
        # Pattern 3: <Timestamp> <Level> <Component>: <Content> (no NodeID)
        re.compile(
            r'(?P<Timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<Level>\w+)\s+'
            r'(?P<Component>[\w\.]+):\s+'
            r'(?P<Content>.*)'
        ),
        # Pattern 4: <NodeID> <Timestamp> <Level> <Component>: <Content>
        re.compile(
            r'(?P<NodeID>\w+)\s+'
            r'(?P<Timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<Level>\w+)\s+'
            r'(?P<Component>[\w\.]+):\s+'
            r'(?P<Content>.*)'
        )
    ]
    
    def __init__(
        self,
        depth: int = 4,
        st: float = 0.4,
        rex: Optional[List[str]] = None
    ):
        """
        Initialize BGL log parser.
        
        Args:
            depth: Drain parser depth
            st: Similarity threshold
            rex: Regular expressions for masking variables
        """
        # Default regex patterns for BGL logs
        if rex is None:
            rex = [
                r"\d+\.\d+\.\d+\.\d+",  # IP addresses
                r"node\d+",              # Node IDs
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}",  # Timestamps
                r"\d+",                  # Numbers
                r"0x[0-9a-fA-F]+",      # Hexadecimal numbers
            ]
        
        self.drain = Drain(depth=depth, st=st, rex=rex)
        
    def parse_line(self, line: str) -> Optional[Dict[str, str]]:
        """
        Parse a single log line using multiple patterns.
        
        Args:
            line: Raw log line
            
        Returns:
            Dictionary with parsed fields or None if parsing fails
        """
        line = line.strip()
        if not line:
            return None
        
        # Try each pattern until one matches
        for pattern in self.BGL_PATTERNS:
            match = pattern.match(line)
            if match:
                parsed = match.groupdict()
                
                # Ensure all fields are present
                if 'NodeID' not in parsed:
                    parsed['NodeID'] = 'unknown'
                
                return parsed
        
        # If no pattern matches, try to extract basic info
        return self._fallback_parse(line)
    
    def _fallback_parse(self, line: str) -> Optional[Dict[str, str]]:
        """
        Fallback parser for unrecognized formats.
        
        Args:
            line: Raw log line
            
        Returns:
            Basic parsed information or None
        """
        # Split by common delimiters
        parts = re.split(r'\s+', line, maxsplit=4)
        
        if len(parts) < 3:
            return None
        
        return {
            'Timestamp': parts[0] if parts[0] else 'unknown',
            'NodeID': parts[1] if len(parts) > 1 else 'unknown',
            'Level': parts[2] if len(parts) > 2 else 'INFO',
            'Component': 'unknown',
            'Content': ' '.join(parts[3:]) if len(parts) > 3 else line
        }
    
    def parse_file(
        self,
        log_file: str,
        output_structured: str,
        output_templates: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse an entire BGL log file.
        
        Args:
            log_file: Path to raw BGL.log file
            output_structured: Path to save structured log CSV
            output_templates: Path to save templates CSV
            
        Returns:
            Tuple of (structured_df, templates_df)
        """
        print(f"Parsing BGL log file: {log_file}")
        
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
                    if failed_lines <= 10:  # Show first 10 failures
                        print(f"  Warning: Could not parse line {line_num}: {line[:100]}...")
                    continue
                
                # Extract event template using Drain
                content = parsed['Content']
                event_id, event_template = self.drain.parse(log_id, content)
                
                # Store structured log entry
                parsed_logs.append({
                    'LineId': log_id,
                    'Timestamp': parsed['Timestamp'],
                    'NodeID': parsed['NodeID'],
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
        
        print(f"\nBGL parsing complete:")
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
        print(f"\nSaving structured BGL log to: {output_structured}")
        structured_df.to_csv(output_structured, index=False)
        
        print(f"Saving BGL templates to: {output_templates}")
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
