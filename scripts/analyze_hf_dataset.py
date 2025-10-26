"""
Analyze AIT dataset on Hugging Face and create visualizations.
This script downloads the dataset, analyzes it, and creates charts.
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_dataset(repo_name: str, output_dir: Path):
    """Analyze dataset from Hugging Face and create visualizations."""
    print("="*70)
    print("ANALYZING HUGGING FACE DATASET")
    print("="*70)
    
    print(f"\nLoading dataset from: {repo_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(repo_name, split='train')
        print(f"✅ Dataset loaded: {len(dataset):,} entries")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    # Convert to DataFrame for analysis
    print("\nConverting to DataFrame...")
    df = pd.DataFrame(dataset)
    print(f"✅ DataFrame created: {len(df):,} rows, {len(df.columns)} columns")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Basic Statistics
    print("\n" + "="*70)
    print("GENERATING BASIC STATISTICS")
    print("="*70)
    
    stats = {
        'total_entries': len(df),
        'columns': list(df.columns),
        'hosts': df['host'].nunique() if 'host' in df.columns else 0,
        'log_types': df['log_type'].nunique() if 'log_type' in df.columns else 0,
        'binary_files': df['is_binary'].sum() if 'is_binary' in df.columns else 0,
        'text_logs': len(df) - (df['is_binary'].sum() if 'is_binary' in df.columns else 0)
    }
    
    # Save statistics
    stats_file = output_dir / 'dataset_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics saved: {stats_file}")
    
    # 2. Host Distribution Chart
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
    
    # 3. Log Type Distribution
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
    
    # 4. Binary vs Text Logs
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
    
    # 5. Content Length Distribution (for text logs)
    if 'content' in df.columns:
        print("\nAnalyzing content length...")
        text_logs = df[~df['is_binary']] if 'is_binary' in df.columns else df
        
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
    
    # 6. File Size Distribution (for binary files)
    if 'file_size_mb' in df.columns:
        print("\nAnalyzing file size distribution...")
        binary_files = df[df['is_binary'] == True]
        
        if len(binary_files) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(binary_files['file_size_mb'], bins=30, edgecolor='black')
            plt.title('Binary File Size Distribution')
            plt.xlabel('File Size (MB)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(output_dir / 'file_size_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: file_size_distribution.png")
    
    # 7. Sample Entries Table
    print("\nCreating sample entries...")
    sample_df = df.head(10)
    sample_file = output_dir / 'sample_entries.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"✅ Saved: sample_entries.csv")
    
    # 8. Anomaly Detection and Analysis
    if 'content' in df.columns:
        print("\nAnalyzing anomalies...")
        
        # Define attack keywords
        attack_keywords = [
            'attack', 'malicious', 'suspicious', 'intrusion', 'breach', 'exploit',
            'unauthorized', 'failed', 'denied', 'blocked', 'firewall', 'scan',
            'nmap', 'sql injection', 'xss', 'csrf', 'ddos', 'botnet', 'trojan',
            'virus', 'malware', 'phishing', 'spam', 'hack', 'crack', 'backdoor',
            'rootkit', 'keylogger', 'ransomware', 'adware', 'spyware', 'password cracking'
        ]
        
        # Add anomaly detection column
        text_logs = df[~df['is_binary']] if 'is_binary' in df.columns else df
        
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
            
            plt.figure(figsize=(10, 6))
            anomaly_by_type.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Anomaly Distribution by Log Type')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(output_dir / 'anomaly_by_log_type.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: anomaly_by_log_type.png")
        
        stats['anomaly_count'] = int(anomaly_count)
        stats['normal_count'] = int(normal_count)
    
    # 9. Comprehensive Summary Report
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    report = f"""
Dataset Analysis Report: {repo_name}
{'='*70}

BASIC STATISTICS
{'-'*70}
Total Entries: {stats['total_entries']:,}
Unique Hosts: {stats['hosts']:,}
Unique Log Types: {stats['log_types']:,}
Text Logs: {stats['text_logs']:,}
Binary Files: {stats['binary_files']:,}
"""
    
    if 'anomaly_count' in stats:
        report += f"""Normal Logs: {stats['normal_count']:,}
Anomaly Logs: {stats['anomaly_count']:,}
Anomaly Rate: {(stats['anomaly_count']/stats['total_entries']*100):.2f}%

COLUMN INFORMATION
{'-'*70}
{', '.join(stats['columns'])}

TOP HOSTS
{'-'*70}
"""
    
    if 'host' in df.columns:
        top_hosts = df['host'].value_counts().head(10)
        for host, count in top_hosts.items():
            report += f"{host}: {count:,}\n"
    
    report += f"\n\nLOG TYPES\n{'-'*70}\n"
    if 'log_type' in df.columns:
        log_types = df['log_type'].value_counts()
        for log_type, count in log_types.items():
            report += f"{log_type}: {count:,}\n"
    
    # Save report
    report_file = output_dir / 'analysis_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"✅ Report saved: {report_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"All outputs saved to: {output_dir}")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Hugging Face dataset and create visualizations')
    parser.add_argument('--repo', required=True, help='Hugging Face repository name')
    parser.add_argument('--output', default='analysis_output', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    analyze_dataset(args.repo, output_dir)


if __name__ == "__main__":
    main()

