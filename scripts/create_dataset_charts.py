"""
Create charts from AIT dataset CSV files.
This is a standalone version that works with local CSV files.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

def create_charts_from_csv(csv_file: Path, output_dir: Path):
    """Create charts from a CSV file."""
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    print("="*70)
    print("CREATING CHARTS FROM CSV")
    print("="*70)
    
    print(f"\nLoading CSV: {csv_file}")
    
    # Read CSV in chunks to handle large files
    chunks = []
    chunk_size = 100000
    
    try:
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunks.append(chunk)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded {len(df):,} entries")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Host Distribution
    if 'host' in df.columns:
        print("\nCreating host distribution chart...")
        host_counts = df['host'].value_counts().head(20)
        
        plt.figure(figsize=(12, 6))
        host_counts.plot(kind='bar')
        plt.title('Top 20 Hosts by Log Entry Count')
        plt.xlabel('Host')
        plt.ylabel('Number of Log Entries')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'host_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: host_distribution.png")
    
    # 2. Log Type Distribution
    if 'log_type' in df.columns:
        print("\nCreating log type distribution chart...")
        log_type_counts = df['log_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        log_type_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Log Type Distribution')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(output_dir / 'log_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: log_type_distribution.png")
    
    # 3. Binary vs Text Logs
    if 'is_binary' in df.columns:
        print("\nCreating binary vs text logs chart...")
        binary_count = df['is_binary'].sum()
        text_count = len(df) - binary_count
        
        plt.figure(figsize=(8, 6))
        plt.bar(['Text Logs', 'Binary Files'], [text_count, binary_count])
        plt.title('Text Logs vs Binary Files')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'binary_vs_text.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: binary_vs_text.png")
    
    # 4. Content Length Distribution
    if 'content' in df.columns and 'is_binary' in df.columns:
        print("\nAnalyzing content length...")
        text_logs = df[~df['is_binary']]
        
        if len(text_logs) > 0:
            text_logs['content_length'] = text_logs['content'].str.len()
            
            plt.figure(figsize=(10, 6))
            plt.hist(text_logs['content_length'], bins=50, edgecolor='black')
            plt.title('Log Content Length Distribution')
            plt.xlabel('Character Count')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(output_dir / 'content_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: content_length_distribution.png")
    
    # 5. Anomaly Detection and Distribution
    if 'content' in df.columns:
        print("\nAnalyzing anomalies...")
        
        # Define attack keywords
        attack_keywords = [
            'attack', 'malicious', 'suspicious', 'intrusion', 'breach', 'exploit',
            'unauthorized', 'failed', 'denied', 'blocked', 'firewall', 'scan',
            'nmap', 'sql injection', 'xss', 'csrf', 'ddos', 'botnet', 'trojan',
            'virus', 'malware', 'phishing', 'spam', 'hack', 'crack', 'backdoor'
        ]
        
        def detect_anomaly(content):
            if pd.isna(content):
                return False
            content_lower = str(content).lower()
            return any(keyword in content_lower for keyword in attack_keywords)
        
        # Add anomaly flag
        df['is_anomaly'] = df['content'].apply(detect_anomaly)
        
        anomaly_count = df['is_anomaly'].sum()
        normal_count = len(df) - anomaly_count
        
        print(f"Normal logs: {normal_count:,}")
        print(f"Anomaly logs: {anomaly_count:,}")
        
        # Create anomaly distribution chart
        plt.figure(figsize=(8, 6))
        plt.bar(['Normal Logs', 'Anomaly Logs'], [normal_count, anomaly_count])
        plt.title('Normal vs Anomaly Log Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: anomaly_distribution.png")
        
        # Anomaly by host
        if 'host' in df.columns:
            anomaly_by_host = df[df['is_anomaly']]['host'].value_counts().head(15)
            
            if len(anomaly_by_host) > 0:
                plt.figure(figsize=(12, 6))
                anomaly_by_host.plot(kind='bar')
                plt.title('Top 15 Hosts by Anomaly Count')
                plt.xlabel('Host')
                plt.ylabel('Anomaly Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(output_dir / 'anomaly_by_host.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved: anomaly_by_host.png")
        
        # Anomaly by log type
        if 'log_type' in df.columns:
            anomaly_by_type = df[df['is_anomaly']]['log_type'].value_counts()
            
            if len(anomaly_by_type) > 0:
                plt.figure(figsize=(10, 6))
                anomaly_by_type.plot(kind='pie', autopct='%1.1f%%')
                plt.title('Anomaly Distribution by Log Type')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(output_dir / 'anomaly_by_log_type.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ Saved: anomaly_by_log_type.png")
    
    # 6. Generate Statistics
    stats = {
        'total_entries': len(df),
        'columns': list(df.columns),
        'hosts': df['host'].nunique() if 'host' in df.columns else 0,
        'log_types': df['log_type'].nunique() if 'log_type' in df.columns else 0,
        'binary_files': df['is_binary'].sum() if 'is_binary' in df.columns else 0,
        'text_logs': len(df) - (df['is_binary'].sum() if 'is_binary' in df.columns else 0)
    }
    
    if 'is_anomaly' in df.columns:
        stats['anomaly_count'] = int(df['is_anomaly'].sum())
        stats['normal_count'] = int(len(df) - stats['anomaly_count'])
    
    # Save statistics
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Statistics saved: {stats_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Entries: {stats['total_entries']:,}")
    print(f"Hosts: {stats['hosts']:,}")
    print(f"Log Types: {stats['log_types']:,}")
    print(f"Text Logs: {stats['text_logs']:,}")
    print(f"Binary Files: {stats['binary_files']:,}")
    
    if 'anomaly_count' in stats:
        print(f"Normal Logs: {stats['normal_count']:,}")
        print(f"Anomaly Logs: {stats['anomaly_count']:,}")
    
    print(f"\n✅ All charts saved to: {output_dir}")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create charts from AIT dataset CSV')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--output', default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv_file)
    output_dir = Path(args.output)
    
    create_charts_from_csv(csv_file, output_dir)


if __name__ == "__main__":
    main()

