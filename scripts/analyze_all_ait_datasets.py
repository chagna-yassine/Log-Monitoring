"""
Comprehensive Analysis Script for AIT Datasets on Hugging Face

This script loads an AIT dataset from Hugging Face, analyzes it thoroughly,
and displays comprehensive statistics including:
- Total number of logs
- Breakdown by host, log type, and other dimensions
- Sample logs from different types and hosts

Updated for single-dataset analysis to avoid memory issues.
"""

import sys
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import json
from collections import defaultdict, Counter
from datetime import datetime
import os
import getpass

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_subheader(title):
    """Print a formatted subheader."""
    print(f"\n{'-'*80}")
    print(f"  {title}")
    print("-"*80)

def load_dataset_with_error_handling(repo_name):
    """Load a dataset from Hugging Face with proper error handling."""
    print(f"Loading: {repo_name}")
    try:
        dataset = load_dataset(repo_name, split='train')
        print(f"✅ Successfully loaded: {len(dataset):,} entries")
        return dataset
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return None

def analyze_dataset(dataset_repo):
    """Analyze a single dataset and provide comprehensive statistics."""
    
    print_header("AIT DATASET ANALYSIS")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_repo}")
    
    # Load dataset
    print_header("LOADING DATASET")
    dataset = load_dataset_with_error_handling(dataset_repo)
    
    if dataset is None:
        print("❌ Dataset could not be loaded. Exiting.")
        return
    
    # Convert to pandas DataFrame for easier analysis
    print("Converting to DataFrame...")
    df = pd.DataFrame(dataset)
    print(f"✅ DataFrame created: {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")
    
    # Basic Statistics
    print_header("BASIC STATISTICS")
    print(f"Dataset: {dataset_repo}")
    print(f"Total Log Entries: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Host Analysis
    if 'host' in df.columns:
        print_subheader("Host Analysis")
        host_counts = df['host'].value_counts()
        print(f"Total Unique Hosts: {len(host_counts)}")
        print(f"\nTop 15 Hosts by Log Count:")
        for host, count in host_counts.head(15).items():
            percentage = (count / len(df)) * 100
            print(f"  {host}: {count:,} ({percentage:.2f}%)")
    
    # Log Type Analysis
    if 'log_type' in df.columns:
        print_subheader("Log Type Analysis")
        log_type_counts = df['log_type'].value_counts()
        print(f"Total Unique Log Types: {len(log_type_counts)}")
        print(f"\nLog Types Distribution:")
        for log_type, count in log_type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {log_type}: {count:,} ({percentage:.2f}%)")
    
    # Binary vs Text Analysis
    if 'is_binary' in df.columns:
        print_subheader("Binary vs Text Logs")
        binary_count = df['is_binary'].sum()
        text_count = len(df) - binary_count
        print(f"Text Logs: {text_count:,}")
        print(f"Binary Files: {binary_count:,}")
        print(f"Binary Percentage: {(binary_count / len(df)) * 100:.2f}%")
    
    # Host vs Log Type Cross Analysis
    if 'host' in df.columns and 'log_type' in df.columns:
        print_subheader("Host-Log Type Cross Analysis")
        cross_tab = pd.crosstab(df['host'], df['log_type'])
        print("\nTop Hosts by Log Type:")
        
        # Find hosts with most variety of log types
        host_log_type_counts = df.groupby('host')['log_type'].nunique().sort_values(ascending=False)
        print(f"\nHosts with Most Log Type Variety:")
        for host, count in host_log_type_counts.head(10).items():
            print(f"  {host}: {count} different log types")
        
        # Show cross-tabulation for top hosts
        print(f"\nDetailed Breakdown (Top 5 Hosts):")
        for host in host_log_type_counts.head(5).index:
            host_data = df[df['host'] == host]
            log_type_breakdown = host_data['log_type'].value_counts()
            print(f"\n  {host}:")
            for log_type, count in log_type_breakdown.items():
                print(f"    {log_type}: {count:,}")
    
    # Sample Logs Analysis
    print_header("SAMPLE LOG ENTRIES")
    
    # Samples by Log Type
    if 'log_type' in df.columns and 'content' in df.columns:
        print_subheader("Sample Logs by Type (First Example from Each Type)")
        log_types = df['log_type'].unique()
        for log_type in sorted(log_types):
            sample = df[df['log_type'] == log_type]
            if len(sample) > 0:
                sample_log = sample.iloc[0]
                print(f"\n📋 Log Type: {log_type}")
                if 'content' in sample_log:
                    content = str(sample_log['content'])
                    # Truncate very long content
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    print(f"   Content: {content}")
                if 'host' in sample_log:
                    print(f"   Host: {sample_log['host']}")
                if 'path' in sample_log:
                    print(f"   Path: {sample_log['path']}")
    
    # Samples by Host
    if 'host' in df.columns and 'content' in df.columns:
        print_subheader("Sample Logs by Host (First Example from Each Host)")
        hosts = df['host'].unique()
        # Show samples from top 10 hosts
        top_hosts = df['host'].value_counts().head(10).index
        for host in top_hosts:
            sample = df[df['host'] == host]
            if len(sample) > 0:
                sample_log = sample.iloc[0]
                print(f"\n🖥️  Host: {host}")
                if 'content' in sample_log:
                    content = str(sample_log['content'])
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    print(f"   Content: {content}")
                if 'log_type' in sample_log:
                    print(f"   Log Type: {sample_log['log_type']}")
                if 'path' in sample_log:
                    print(f"   Path: {sample_log['path']}")
    
    # Multiple samples from different types and hosts
    print_subheader("Multiple Random Samples from Different Types and Hosts")
    print("\n🎲 Random Sampling (5 examples):")
    
    # Get samples ensuring diversity in both type and host
    samples_shown = set()
    for i in range(5):
        sample = df.sample(n=1).iloc[0]
        sample_key = (sample['log_type'] if 'log_type' in sample else 'unknown', 
                     sample['host'] if 'host' in sample else 'unknown')
        
        if sample_key not in samples_shown:
            samples_shown.add(sample_key)
            print(f"\n  Sample {i+1}:")
            if 'host' in sample:
                print(f"    Host: {sample['host']}")
            if 'log_type' in sample:
                print(f"    Type: {sample['log_type']}")
            if 'content' in sample:
                content = str(sample['content'])
                if len(content) > 400:
                    content = content[:400] + "... [truncated]"
                print(f"    Content: {content}")
            if 'path' in sample:
                print(f"    Path: {sample['path']}")
    
    # Content Analysis (for text logs)
    if 'content' in df.columns:
        print_subheader("Content Analysis")
        text_logs = df[~df['is_binary']] if 'is_binary' in df.columns else df
        
        if len(text_logs) > 0:
            # Add content length column
            text_logs['content_length'] = text_logs['content'].astype(str).str.len()
            
            print(f"Text Logs Analyzed: {len(text_logs):,}")
            print(f"Average Content Length: {text_logs['content_length'].mean():.1f} characters")
            print(f"Median Content Length: {text_logs['content_length'].median():.1f} characters")
            print(f"Min Content Length: {text_logs['content_length'].min()} characters")
            print(f"Max Content Length: {text_logs['content_length'].max()} characters")
            
            # Show distribution percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            print(f"\nContent Length Percentiles:")
            for p in percentiles:
                value = text_logs['content_length'].quantile(p/100)
                print(f"  {p}th percentile: {value:.1f} characters")
    
    # File Size Analysis (for binary logs)
    if 'file_size_mb' in df.columns:
        print_subheader("Binary File Size Analysis")
        binary_files = df[df['is_binary'] == True]
        
        if len(binary_files) > 0:
            print(f"Binary Files: {len(binary_files):,}")
            print(f"Average Size: {binary_files['file_size_mb'].mean():.2f} MB")
            print(f"Median Size: {binary_files['file_size_mb'].median():.2f} MB")
            print(f"Total Size: {binary_files['file_size_mb'].sum():.2f} MB")
    
    # Temporal Analysis (if timestamps available)
    if 'timestamp' in df.columns or 'time' in df.columns:
        print_subheader("Temporal Analysis")
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        print(f"Timestamp column: {timestamp_col}")
        print(f"Note: Detailed temporal analysis requires parsing timestamps")
    
    # Summary Statistics Dictionary
    summary_stats = {
        'analysis_timestamp': datetime.now().isoformat(),
        'dataset': dataset_repo,
        'total_logs': int(len(df)),
        'columns': list(df.columns),
        'num_hosts': int(df['host'].nunique()) if 'host' in df.columns else 0,
        'num_log_types': int(df['log_type'].nunique()) if 'log_type' in df.columns else 0,
        'num_text_logs': int(len(df) - df['is_binary'].sum()) if 'is_binary' in df.columns else len(df),
        'num_binary_files': int(df['is_binary'].sum()) if 'is_binary' in df.columns else 0,
    }
    
    # Add host distribution
    if 'host' in df.columns:
        summary_stats['host_distribution'] = df['host'].value_counts().to_dict()
    
    # Add log type distribution
    if 'log_type' in df.columns:
        summary_stats['log_type_distribution'] = df['log_type'].value_counts().to_dict()
    
    print_header("ANALYSIS COMPLETE")
    print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save summary to file
    dataset_name = dataset_repo.split('/')[-1]
    output_file = Path(f'{dataset_name}_analysis_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"\n✅ Summary saved to: {output_file}")
    
    return summary_stats

def main():
    """Main function."""
    import argparse
    
    # Available datasets
    available_datasets = [
        'chYassine/ait-wilson-raw-v01',
        'chYassine/ait-wheeler-raw-v01',
        'chYassine/ait-wardbeck-raw-v01',
        'chYassine/ait-shaw-raw-v01',
        'chYassine/ait-santos-raw-v01',
        'chYassine/air-russellmitchell-raw-v01',
        'chYassine/ait-harrison-raw-v01',
        'chYassine/ait-fox-raw-v02'
    ]
    
    parser = argparse.ArgumentParser(
        description='Comprehensive analysis of a single AIT dataset on Hugging Face',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Prevent abbreviations that might conflict with kernel args
        epilog="""
Examples:
  # Analyze a specific dataset
  python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02
  
  # With HuggingFace token
  python scripts/analyze_all_ait_datasets.py --repo chYassine/ait-fox-raw-v02 --token
  
  # List available datasets
  python scripts/analyze_all_ait_datasets.py --list
        """
    )
    parser.add_argument('--repo', type=str, required=False,
                       help='HuggingFace repository name (e.g., chYassine/ait-fox-raw-v02)')
    parser.add_argument('--token', action='store_true',
                       help='Prompt for HuggingFace token (for private datasets)')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets and exit')
    
    try:
        args = parser.parse_args()
    except SystemExit:
        # Jupyter/Colab kernel passes additional args that cause SystemExit
        # Prompt for dataset interactively
        import types
        args = types.SimpleNamespace()
        args.repo = None
        args.token = False
        args.list = False
    
    if args.list:
        print("Available AIT Datasets:")
        for i, repo in enumerate(available_datasets, 1):
            print(f"  {i}. {repo}")
        return
    
    # Handle HuggingFace authentication
    if args.token or os.environ.get('HUGGINGFACE_HUB_TOKEN') is None:
        print("\n🔑 HuggingFace Authentication")
        print("If your dataset is private, you need to provide a token.")
        
        # Check if we're in an interactive environment
        try:
            if args.token:
                token = getpass.getpass("Enter your HuggingFace token (input hidden): ")
                if token:
                    from huggingface_hub import login
                    login(token=token)
                    print("✅ Logged in to HuggingFace")
        except Exception as e:
            print(f"⚠️ Could not authenticate: {e}")
            print("Continuing without authentication...")
    
    # Get repository to analyze
    repo = args.repo
    if not repo:
        # Interactive prompt
        print("\n📊 Available Datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i}. {dataset}")
        
        try:
            choice = input("\nEnter dataset number or full repository name: ").strip()
            
            # Check if it's a number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_datasets):
                    repo = available_datasets[idx]
                else:
                    print("❌ Invalid choice")
                    return
            else:
                repo = choice
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled by user")
            return
    
    if not repo:
        print("❌ No dataset specified. Use --repo or run interactively.")
        return
    
    # Analyze the dataset
    print(f"\n🚀 Starting analysis of: {repo}")
    analyze_dataset(repo)

if __name__ == "__main__":
    main()

