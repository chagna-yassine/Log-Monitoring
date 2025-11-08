#!/usr/bin/env python3
"""
Example script showing how to analyze the extracted AIT attack logs

Usage:
    python analyze_ait_attacks.py fox_attacks.parquet
    python analyze_ait_attacks.py fox_attacks.csv
"""

import sys
import pandas as pd
import json
from collections import Counter


def load_attacks(file_path: str) -> pd.DataFrame:
    """Load attack logs from CSV or Parquet file"""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("File must be .csv or .parquet")


def analyze_attacks(df: pd.DataFrame) -> None:
    """Perform comprehensive analysis of attack logs"""
    
    print("=" * 80)
    print("📊 AIT Attack Log Analysis")
    print("=" * 80)
    
    # Basic statistics
    print("\n1️⃣ Basic Statistics")
    print(f"   Total attack log lines: {len(df):,}")
    print(f"   Unique log files: {df['log_file'].nunique()}")
    print(f"   Unique hosts: {df['host'].nunique()}")
    print(f"   Log types: {df['log_type'].nunique()}")
    
    # Attack label distribution
    print("\n2️⃣ Attack Label Distribution")
    all_labels = []
    for labels in df['attack_labels'].dropna():
        all_labels.extend(labels.split(','))
    
    label_counts = Counter(all_labels)
    print(f"   Unique attack labels: {len(label_counts)}")
    print(f"\n   Top 10 Attack Types:")
    for label, count in label_counts.most_common(10):
        percentage = (count / len(df)) * 100
        print(f"   - {label:30s}: {count:6,} ({percentage:5.2f}%)")
    
    # Attacks by host
    print("\n3️⃣ Attacks by Host")
    attacks_by_host = df.groupby('host').size().sort_values(ascending=False)
    print(f"   Total hosts with attack logs: {len(attacks_by_host)}")
    print(f"\n   Top 10 Most Attacked Hosts:")
    for host, count in attacks_by_host.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"   - {host:30s}: {count:6,} ({percentage:5.2f}%)")
    
    # Attacks by log type
    print("\n4️⃣ Attacks by Log Type")
    attacks_by_type = df.groupby('log_type').size().sort_values(ascending=False)
    print(f"\n   Top 10 Log Types with Most Attack Events:")
    for log_type, count in attacks_by_type.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"   - {log_type:30s}: {count:6,} ({percentage:5.2f}%)")
    
    # Attack chain analysis (co-occurring labels)
    print("\n5️⃣ Attack Chain Analysis (Co-occurring Labels)")
    multi_label_rows = df[df['attack_labels'].str.contains(',', na=False)]
    print(f"   Log lines with multiple labels: {len(multi_label_rows):,}")
    
    if len(multi_label_rows) > 0:
        label_combinations = Counter(multi_label_rows['attack_labels'])
        print(f"\n   Top 10 Label Combinations:")
        for combo, count in label_combinations.most_common(10):
            print(f"   - {combo:50s}: {count:5,}")
    
    # Timeline analysis (if timestamp available)
    print("\n6️⃣ Timeline Analysis")
    non_empty_timestamps = df['timestamp'].dropna()
    if len(non_empty_timestamps) > 0:
        print(f"   Log lines with timestamps: {len(non_empty_timestamps):,}")
        print(f"   First attack log: {non_empty_timestamps.iloc[0]}")
        print(f"   Last attack log: {non_empty_timestamps.iloc[-1]}")
    else:
        print("   No timestamp information available")
    
    # Sample attack logs
    print("\n7️⃣ Sample Attack Logs")
    print("\n   First 3 Attack Log Entries:")
    for idx, row in df.head(3).iterrows():
        print(f"\n   [{idx}] {row['log_file']}")
        print(f"       Host: {row['host']}")
        print(f"       Labels: {row['attack_labels']}")
        print(f"       Content: {row['log_content'][:100]}...")
    
    # Attack severity analysis (based on attack type)
    print("\n8️⃣ Attack Severity Analysis")
    high_severity = ['rce', 'escalate', 'data_exfiltration', 'webshell_upload']
    medium_severity = ['password_cracking', 'attacker_change_user']
    low_severity = ['scan']
    
    high_count = sum(1 for labels in df['attack_labels'] if any(h in labels for h in high_severity))
    medium_count = sum(1 for labels in df['attack_labels'] if any(m in labels for m in medium_severity))
    low_count = sum(1 for labels in df['attack_labels'] if any(l in labels for l in low_severity))
    
    print(f"   🔴 High Severity (RCE, Escalation, Exfiltration): {high_count:,} ({(high_count/len(df)*100):.2f}%)")
    print(f"   🟡 Medium Severity (Password Cracking, User Switch): {medium_count:,} ({(medium_count/len(df)*100):.2f}%)")
    print(f"   🟢 Low Severity (Scanning): {low_count:,} ({(low_count/len(df)*100):.2f}%)")
    
    print("\n" + "=" * 80)


def export_filtered_attacks(df: pd.DataFrame, attack_type: str, output_file: str) -> None:
    """Export logs of a specific attack type"""
    filtered = df[df['attack_labels'].str.contains(attack_type, na=False)]
    
    if len(filtered) == 0:
        print(f"⚠️  No logs found with attack type: {attack_type}")
        return
    
    filtered.to_csv(output_file, index=False)
    print(f"✅ Exported {len(filtered)} logs with '{attack_type}' to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ait_attacks.py <attack_log_file.parquet|csv>")
        print("\nExample:")
        print("  python analyze_ait_attacks.py fox_attacks.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        print(f"📂 Loading attack logs from {file_path}...")
        df = load_attacks(file_path)
        print(f"✅ Loaded {len(df):,} attack log entries")
        
        # Run analysis
        analyze_attacks(df)
        
        # Optional: Export specific attack types
        print("\n💾 Exporting Filtered Attack Types...")
        export_filtered_attacks(df, 'escalate', 'escalation_attacks.csv')
        export_filtered_attacks(df, 'scan', 'scan_attacks.csv')
        export_filtered_attacks(df, 'rce', 'rce_attacks.csv')
        
        print("\n✨ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

