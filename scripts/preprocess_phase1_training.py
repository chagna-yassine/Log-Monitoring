"""
First Training Phase Preprocessing for AIT Log Understanding Model

This script will prepare the AIT dataset from HuggingFace for the first training phase,
which builds a log understanding base model with:
- Masked log modeling
- Next-log prediction
- Contrastive session summarization
- Template-ID classification

The preprocessing includes:
- Context-aware session grouping (by host and time window)
- Session metadata extraction
- Curriculum learning structure (sequence length buckets)
- Multiple training objective preparation
"""

import sys
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
from datasets import load_dataset, Dataset, Features, Value, Sequence
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase1Preprocessor:
    """
    Preprocesses AIT dataset for first training phase.
    
    Creates session-level data with:
    - Context-aware grouping (host + time window)
    - Session metadata (host, log_type, timestamps, length buckets)
    - Prepared for multiple training objectives
    - Curriculum learning structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ait_config = config.get('ait_dataset', {})
        self.output_config = config.get('output', {}).get('ait', {})
        
        # Setup paths
        self.base_path = Path(self.ait_config.get('base_path', 'datasets/ait'))
        self.dataset_name = self.ait_config.get('selected_dataset', 'fox')
        self.output_path = Path(self.output_config.get('base_path', 'datasets/ait/output/ait'))
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Phase 1 specific output path
        self.phase1_output = self.output_path / 'phase1_training'
        self.phase1_output.mkdir(parents=True, exist_ok=True)
        
        # Training phase configuration
        self.phase1_config = config.get('phase1_training', {
            'time_window_minutes': 30,
            'max_sequence_length': 512,
            'min_sequence_length': 4,
            'curriculum_buckets': [8, 16, 32, 64, 128, 256, 512],
            'mask_probability': 0.15,
            'next_log_window_size': 5,
            'contrastive_augmentations': ['time_crop', 'log_type_sample']
        })
        
        logger.info(f"Phase1 Preprocessor initialized for dataset: {self.dataset_name}")
        logger.info(f"Output path: {self.phase1_output}")
    
    def load_data_from_hf(self, repo_name: str) -> pd.DataFrame:
        """
        Load AIT dataset from HuggingFace.
        
        Args:
            repo_name: HuggingFace repository name
            
        Returns:
            DataFrame with log entries
        """
        logger.info(f"Loading dataset from HuggingFace: {repo_name}")
        
        try:
            dataset = load_dataset(repo_name, split='train')
            logger.info(f"✅ Loaded {len(dataset):,} entries")
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset)
            logger.info(f"✅ Converted to DataFrame: {len(df):,} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise
    
    def load_data_from_local(self) -> pd.DataFrame:
        """
        Load processed AIT data from local files.
        
        Returns:
            DataFrame with structured log entries
        """
        logger.info("Loading data from local processed files...")
        
        structured_path = self.output_path / self.output_config.get('structured_log', 'ait_structured.csv')
        
        if not structured_path.exists():
            raise FileNotFoundError(
                f"Structured log file not found: {structured_path}\n"
                f"Please run preprocessing first: python scripts/preprocess_ait.py"
            )
        
        df = pd.read_csv(structured_path)
        logger.info(f"✅ Loaded {len(df):,} structured log entries from {structured_path}")
        
        return df
    
    def create_session_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create context-aware session groups.
        
        Groups logs by:
        - Host (for host-specific patterns)
        - Time window (for temporal relationships)
        - Log type (for domain specialization)
        
        Args:
            df: DataFrame with log entries
            
        Returns:
            DataFrame with session groups
        """
        logger.info("Creating context-aware session groups...")
        
        # Determine grouping columns
        has_host = 'host' in df.columns or 'Host' in df.columns
        has_timestamp = 'timestamp' in df.columns or 'Timestamp' in df.columns
        has_log_type = 'log_type' in df.columns or 'LogType' in df.columns
        
        # Normalize column names
        host_col = 'host' if 'host' in df.columns else ('Host' if 'Host' in df.columns else None)
        timestamp_col = 'timestamp' if 'timestamp' in df.columns else ('Timestamp' if 'Timestamp' in df.columns else None)
        log_type_col = 'log_type' if 'log_type' in df.columns else ('LogType' if 'LogType' in df.columns else None)
        
        time_window_minutes = self.phase1_config['time_window_minutes']
        
        # Create session IDs
        session_ids = []
        session_metadata = []
        
        for idx, row in df.iterrows():
            # Extract components
            host = str(row[host_col]) if host_col and pd.notna(row[host_col]) else 'unknown'
            
            # Parse timestamp and create time window
            time_window = 'unknown'
            if timestamp_col and pd.notna(row[timestamp_col]):
                timestamp = row[timestamp_col]
                try:
                    if isinstance(timestamp, str):
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d %H:%M:%S', '%b %d %H:%M:%S']:
                            try:
                                dt = datetime.strptime(timestamp, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            dt = datetime.now()
                    else:
                        dt = timestamp
                    
                    # Create time window (round down to nearest window)
                    window_start = dt.replace(minute=(dt.minute // time_window_minutes) * time_window_minutes, 
                                            second=0, microsecond=0)
                    time_window = window_start.strftime('%Y-%m-%d_%H:%M')
                except:
                    time_window = 'unknown'
            
            log_type = str(row[log_type_col]) if log_type_col and pd.notna(row[log_type_col]) else 'unknown'
            
            # Create session ID: host_logtype_timewindow
            session_id = f"{host}_{log_type}_{time_window}"
            session_ids.append(session_id)
            
            # Store metadata
            session_metadata.append({
                'host': host,
                'log_type': log_type,
                'time_window': time_window,
                'timestamp': timestamp if timestamp_col and pd.notna(row[timestamp_col]) else None
            })
            
            if (idx + 1) % 100000 == 0:
                logger.info(f"  Processed {idx + 1:,} entries, created {len(set(session_ids)):,} unique sessions")
        
        # Add session information to dataframe
        df = df.copy()
        df['session_id'] = session_ids
        df['session_host'] = [m['host'] for m in session_metadata]
        df['session_log_type'] = [m['log_type'] for m in session_metadata]
        df['session_time_window'] = [m['time_window'] for m in session_metadata]
        
        # Get template ID if available
        if 'EventId' in df.columns:
            df['template_id'] = df['EventId'].fillna(0).astype(int)
        elif 'event_id' in df.columns:
            df['template_id'] = df['event_id'].fillna(0).astype(int)
        else:
            logger.warning("No EventId/event_id column found, setting template_id to 0")
            df['template_id'] = 0
        
        # Get content/text
        if 'content' in df.columns:
            df['log_text'] = df['content'].astype(str)
        elif 'Content' in df.columns:
            df['log_text'] = df['Content'].astype(str)
        elif 'TextSequence' in df.columns:
            df['log_text'] = df['TextSequence'].astype(str)
        else:
            logger.warning("No content/text column found")
            df['log_text'] = ''
        
        logger.info(f"✅ Created {df['session_id'].nunique():,} unique sessions")
        logger.info(f"  Sessions per host: {df.groupby('session_host')['session_id'].nunique().mean():.1f}")
        logger.info(f"  Sessions per log type: {df.groupby('session_log_type')['session_id'].nunique().mean():.1f}")
        
        return df
    
    def build_session_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build sequences from session groups.
        
        Args:
            df: DataFrame with session groups
            
        Returns:
            DataFrame with session sequences
        """
        logger.info("Building session sequences...")
        
        # Group by session
        session_sequences = []
        
        for session_id, group in df.groupby('session_id'):
            # Sort by timestamp if available
            if 'timestamp' in group.columns or 'Timestamp' in group.columns:
                timestamp_col = 'timestamp' if 'timestamp' in group.columns else 'Timestamp'
                group = group.sort_values(timestamp_col)
            
            # Extract sequence components
            template_ids = group['template_id'].tolist()
            log_texts = group['log_text'].tolist()
            
            # Get session metadata
            session_host = group['session_host'].iloc[0]
            session_log_type = group['session_log_type'].iloc[0]
            session_time_window = group['session_time_window'].iloc[0]
            
            # Get labels if available
            label = 0
            is_attack = False
            attack_labels = []
            
            if 'Label' in group.columns:
                label = int(group['Label'].max())  # Session is attack if any entry is attack
            elif 'label' in group.columns:
                label = int(group['label'].max())
            
            if 'IsAttack' in group.columns:
                is_attack = bool(group['IsAttack'].max())
            elif 'is_attack' in group.columns:
                is_attack = bool(group['is_attack'].max())
            
            if 'AttackLabels' in group.columns:
                attack_labels_raw = group['AttackLabels'].dropna().tolist()
                if attack_labels_raw:
                    attack_labels = attack_labels_raw
            
            sequence_length = len(template_ids)
            
            # Only include sequences within length limits
            if sequence_length < self.phase1_config['min_sequence_length']:
                continue
            
            # Determine curriculum bucket
            curriculum_bucket = self._get_curriculum_bucket(sequence_length)
            
            session_sequences.append({
                'session_id': session_id,
                'template_ids': template_ids,
                'log_texts': log_texts,
                'sequence_length': sequence_length,
                'curriculum_bucket': curriculum_bucket,
                'host': session_host,
                'log_type': session_log_type,
                'time_window': session_time_window,
                'label': label,
                'is_attack': is_attack,
                'attack_labels': attack_labels
            })
            
            if len(session_sequences) % 10000 == 0:
                logger.info(f"  Built {len(session_sequences):,} session sequences")
        
        sessions_df = pd.DataFrame(session_sequences)
        
        logger.info(f"✅ Built {len(sessions_df):,} session sequences")
        logger.info(f"  Average sequence length: {sessions_df['sequence_length'].mean():.2f}")
        logger.info(f"  Min/Max length: {sessions_df['sequence_length'].min()}/{sessions_df['sequence_length'].max()}")
        logger.info(f"  Curriculum buckets distribution:")
        for bucket in sorted(sessions_df['curriculum_bucket'].unique()):
            count = len(sessions_df[sessions_df['curriculum_bucket'] == bucket])
            logger.info(f"    {bucket}: {count:,} sessions")
        
        return sessions_df
    
    def _get_curriculum_bucket(self, length: int) -> int:
        """Get curriculum learning bucket for sequence length."""
        buckets = sorted(self.phase1_config['curriculum_buckets'])
        for bucket in buckets:
            if length <= bucket:
                return bucket
        return buckets[-1]
    
    def prepare_training_data(self, sessions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Prepare data for multiple training objectives.
        
        Creates:
        1. Masked log modeling: sequences with mask positions
        2. Next-log prediction: sliding windows with next token labels
        3. Contrastive session summarization: multiple views of sessions
        4. Template-ID classification: template ID labels
        
        Args:
            sessions_df: DataFrame with session sequences
            
        Returns:
            Dictionary with prepared training data
        """
        logger.info("Preparing training data for multiple objectives...")
        
        training_data = {
            'sessions': [],
            'masked_lm': [],
            'next_log': [],
            'contrastive': [],
            'template_classification': []
        }
        
        for idx, row in sessions_df.iterrows():
            session_id = row['session_id']
            template_ids = row['template_ids']
            log_texts = row['log_texts']
            sequence_length = row['sequence_length']
            
            # Basic session data
            training_data['sessions'].append({
                'session_id': session_id,
                'template_ids': template_ids,
                'log_texts': log_texts,
                'sequence_length': sequence_length,
                'curriculum_bucket': row['curriculum_bucket'],
                'host': row['host'],
                'log_type': row['log_type'],
                'time_window': row['time_window'],
                'label': row['label'],
                'is_attack': row['is_attack']
            })
            
            # 1. Masked Log Modeling
            masked_positions = self._generate_mask_positions(sequence_length)
            training_data['masked_lm'].append({
                'session_id': session_id,
                'template_ids': template_ids,
                'masked_positions': masked_positions,
                'num_masked': len(masked_positions)
            })
            
            # 2. Next-Log Prediction (sliding windows)
            next_log_pairs = self._generate_next_log_pairs(template_ids, log_texts)
            for pair in next_log_pairs:
                training_data['next_log'].append({
                    'session_id': session_id,
                    'context_ids': pair['context'],
                    'context_texts': pair['context_texts'],
                    'next_template_id': pair['next_id'],
                    'next_text': pair['next_text']
                })
            
            # 3. Template-ID Classification
            for i, template_id in enumerate(template_ids):
                training_data['template_classification'].append({
                    'session_id': session_id,
                    'log_text': log_texts[i] if i < len(log_texts) else '',
                    'template_id': template_id,
                    'position': i
                })
            
            # 4. Contrastive Session Summarization (sample for efficiency)
            if idx % 10 == 0:  # Sample 10% for contrastive
                contrastive_views = self._generate_contrastive_views(template_ids, log_texts)
                for view_pair in contrastive_views:
                    training_data['contrastive'].append({
                        'session_id': session_id,
                        'view1_ids': view_pair['view1_ids'],
                        'view1_texts': view_pair['view1_texts'],
                        'view2_ids': view_pair['view2_ids'],
                        'view2_texts': view_pair['view2_texts']
                    })
            
            if (idx + 1) % 10000 == 0:
                logger.info(f"  Prepared {idx + 1:,} sessions")
        
        logger.info("✅ Training data preparation complete")
        logger.info(f"  Sessions: {len(training_data['sessions']):,}")
        logger.info(f"  Masked LM samples: {len(training_data['masked_lm']):,}")
        logger.info(f"  Next-log samples: {len(training_data['next_log']):,}")
        logger.info(f"  Contrastive samples: {len(training_data['contrastive']):,}")
        logger.info(f"  Template classification samples: {len(training_data['template_classification']):,}")
        
        return training_data
    
    def _generate_mask_positions(self, length: int) -> List[int]:
        """Generate random mask positions for masked log modeling."""
        mask_prob = self.phase1_config['mask_probability']
        num_masked = max(1, int(length * mask_prob))
        positions = np.random.choice(length, size=min(num_masked, length), replace=False).tolist()
        return sorted(positions)
    
    def _generate_next_log_pairs(self, template_ids: List[int], log_texts: List[str]) -> List[Dict]:
        """Generate next-log prediction pairs using sliding window."""
        window_size = self.phase1_config['next_log_window_size']
        pairs = []
        
        for i in range(len(template_ids) - 1):
            context = template_ids[max(0, i - window_size + 1):i + 1]
            context_texts = log_texts[max(0, i - window_size + 1):i + 1]
            next_id = template_ids[i + 1]
            next_text = log_texts[i + 1] if i + 1 < len(log_texts) else ''
            
            pairs.append({
                'context': context,
                'context_texts': context_texts,
                'next_id': next_id,
                'next_text': next_text
            })
        
        return pairs
    
    def _generate_contrastive_views(self, template_ids: List[int], log_texts: List[str]) -> List[Dict]:
        """Generate multiple views of the same session for contrastive learning."""
        views = []
        
        # View 1: Time crop (first half)
        # View 2: Time crop (second half)
        if len(template_ids) >= 4:
            mid = len(template_ids) // 2
            views.append({
                'view1_ids': template_ids[:mid],
                'view1_texts': log_texts[:mid],
                'view2_ids': template_ids[mid:],
                'view2_texts': log_texts[mid:]
            })
        
        # View 1: Full sequence
        # View 2: Sampled sequence (every other element)
        if len(template_ids) >= 4:
            sampled_ids = template_ids[::2]
            sampled_texts = [log_texts[i] for i in range(0, len(log_texts), 2)]
            views.append({
                'view1_ids': template_ids,
                'view1_texts': log_texts,
                'view2_ids': sampled_ids,
                'view2_texts': sampled_texts
            })
        
        return views
    
    def save_training_data(self, training_data: Dict[str, Any]) -> None:
        """Save prepared training data to disk."""
        logger.info("Saving training data...")
        
        # Convert lists to JSON strings for CSV compatibility
        def prepare_df_for_csv(data_list):
            df = pd.DataFrame(data_list)
            # Convert list columns to JSON strings
            for col in df.columns:
                if df[col].dtype == object:
                    # Check if column contains lists
                    sample_val = df[col].iloc[0] if len(df) > 0 else None
                    if isinstance(sample_val, list):
                        df[col] = df[col].apply(json.dumps)
            return df
        
        # Save sessions
        sessions_df = prepare_df_for_csv(training_data['sessions'])
        sessions_path = self.phase1_output / 'sessions.csv'
        sessions_df.to_csv(sessions_path, index=False)
        logger.info(f"✅ Saved {len(sessions_df):,} sessions to {sessions_path}")
        
        # Save masked LM data
        masked_lm_df = prepare_df_for_csv(training_data['masked_lm'])
        masked_lm_path = self.phase1_output / 'masked_lm.csv'
        masked_lm_df.to_csv(masked_lm_path, index=False)
        logger.info(f"✅ Saved {len(masked_lm_df):,} masked LM samples to {masked_lm_path}")
        
        # Save next-log data
        next_log_df = prepare_df_for_csv(training_data['next_log'])
        next_log_path = self.phase1_output / 'next_log.csv'
        next_log_df.to_csv(next_log_path, index=False)
        logger.info(f"✅ Saved {len(next_log_df):,} next-log samples to {next_log_path}")
        
        # Save contrastive data
        contrastive_df = prepare_df_for_csv(training_data['contrastive'])
        contrastive_path = self.phase1_output / 'contrastive.csv'
        contrastive_df.to_csv(contrastive_path, index=False)
        logger.info(f"✅ Saved {len(contrastive_df):,} contrastive samples to {contrastive_path}")
        
        # Save template classification data
        template_cls_df = prepare_df_for_csv(training_data['template_classification'])
        template_cls_path = self.phase1_output / 'template_classification.csv'
        template_cls_df.to_csv(template_cls_path, index=False)
        logger.info(f"✅ Saved {len(template_cls_df):,} template classification samples to {template_cls_path}")
        
        # Save metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'total_sessions': len(training_data['sessions']),
            'masked_lm_samples': len(training_data['masked_lm']),
            'next_log_samples': len(training_data['next_log']),
            'contrastive_samples': len(training_data['contrastive']),
            'template_classification_samples': len(training_data['template_classification']),
            'curriculum_buckets': self.phase1_config['curriculum_buckets'],
            'time_window_minutes': self.phase1_config['time_window_minutes'],
            'configuration': self.phase1_config
        }
        
        metadata_path = self.phase1_output / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved metadata to {metadata_path}")
    
    def create_hf_dataset(self, training_data: Dict[str, Any]) -> None:
        """Create HuggingFace datasets for efficient training."""
        logger.info("Creating HuggingFace datasets...")
        
        # Create sessions dataset
        sessions_ds = Dataset.from_pandas(pd.DataFrame(training_data['sessions']))
        sessions_ds.save_to_disk(str(self.phase1_output / 'hf_sessions'))
        logger.info(f"✅ Saved HuggingFace sessions dataset")
        
        # Create masked LM dataset
        masked_lm_ds = Dataset.from_pandas(pd.DataFrame(training_data['masked_lm']))
        masked_lm_ds.save_to_disk(str(self.phase1_output / 'hf_masked_lm'))
        logger.info(f"✅ Saved HuggingFace masked LM dataset")
        
        # Create next-log dataset
        next_log_ds = Dataset.from_pandas(pd.DataFrame(training_data['next_log']))
        next_log_ds.save_to_disk(str(self.phase1_output / 'hf_next_log'))
        logger.info(f"✅ Saved HuggingFace next-log dataset")
        
        # Create contrastive dataset
        contrastive_ds = Dataset.from_pandas(pd.DataFrame(training_data['contrastive']))
        contrastive_ds.save_to_disk(str(self.phase1_output / 'hf_contrastive'))
        logger.info(f"✅ Saved HuggingFace contrastive dataset")
        
        # Create template classification dataset
        template_cls_ds = Dataset.from_pandas(pd.DataFrame(training_data['template_classification']))
        template_cls_ds.save_to_disk(str(self.phase1_output / 'hf_template_classification'))
        logger.info(f"✅ Saved HuggingFace template classification dataset")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main function to run Phase 1 preprocessing."""
    print("="*70)
    print("PHASE 1 TRAINING PREPROCESSING")
    print("="*70)
    
    # Load configuration
    config = load_config()
    
    # Check for Phase 1 config
    if 'phase1_training' not in config:
        # Add default Phase 1 config
        config['phase1_training'] = {
            'time_window_minutes': 30,
            'max_sequence_length': 512,
            'min_sequence_length': 4,
            'curriculum_buckets': [8, 16, 32, 64, 128, 256, 512],
            'mask_probability': 0.15,
            'next_log_window_size': 5,
            'contrastive_augmentations': ['time_crop', 'log_type_sample']
        }
        logger.info("Using default Phase 1 training configuration")
    
    # Initialize preprocessor
    preprocessor = Phase1Preprocessor(config)
    
    # Determine data source
    use_hf = False
    hf_repo = None
    
    if len(sys.argv) > 1:
        hf_repo = sys.argv[1]
        use_hf = True
        logger.info(f"Using HuggingFace dataset: {hf_repo}")
    else:
        logger.info("Using local processed data")
    
    # Load data
    try:
        if use_hf:
            df = preprocessor.load_data_from_hf(hf_repo)
        else:
            df = preprocessor.load_data_from_local()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Create session groups
    df = preprocessor.create_session_groups(df)
    
    # Build session sequences
    sessions_df = preprocessor.build_session_sequences(df)
    
    # Prepare training data
    training_data = preprocessor.prepare_training_data(sessions_df)
    
    # Save training data
    preprocessor.save_training_data(training_data)
    
    # Create HuggingFace datasets
    try:
        preprocessor.create_hf_dataset(training_data)
    except Exception as e:
        logger.warning(f"Failed to create HuggingFace datasets: {e}")
        logger.info("CSV files are still available")
    
    print("\n" + "="*70)
    print("PHASE 1 PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {preprocessor.phase1_output}")
    print("\nGenerated files:")
    print("  - sessions.csv: Session-level data with metadata")
    print("  - masked_lm.csv: Masked log modeling samples")
    print("  - next_log.csv: Next-log prediction samples")
    print("  - contrastive.csv: Contrastive session summarization samples")
    print("  - template_classification.csv: Template-ID classification samples")
    print("  - metadata.json: Processing metadata and statistics")
    print("\nHuggingFace datasets (if created):")
    print("  - hf_sessions/")
    print("  - hf_masked_lm/")
    print("  - hf_next_log/")
    print("  - hf_contrastive/")
    print("  - hf_template_classification/")


if __name__ == "__main__":
    main()

