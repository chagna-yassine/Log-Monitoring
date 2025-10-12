"""
Template Mapper

Creates numerical mapping for event templates.
Maps event templates to sequential numbers based on frequency.
"""

import json
import pandas as pd
from typing import Dict
from pathlib import Path


class TemplateMapper:
    """
    Maps event templates to sequential numerical IDs.
    
    Templates are sorted by frequency and assigned numbers 1, 2, 3, ...
    This creates a consistent mapping for converting sequences.
    """
    
    def __init__(self):
        self.template_to_number = {}
        self.number_to_template = {}
        
    def create_mapping(
        self,
        templates_csv: str,
        output_json: str
    ) -> Dict[int, int]:
        """
        Create numerical mapping from templates CSV.
        
        Args:
            templates_csv: Path to HDFS.log_templates.csv
            output_json: Path to save mapping JSON
            
        Returns:
            Dictionary mapping EventId to template number
        """
        print(f"\nCreating template mapping from: {templates_csv}")
        
        # Load templates (already sorted by frequency)
        templates_df = pd.read_csv(templates_csv)
        
        # Create mapping: EventId -> Sequential Number
        event_id_to_number = {}
        number_to_template = {}
        template_to_number = {}
        
        for idx, row in templates_df.iterrows():
            event_id = int(row['EventId'])
            template = row['EventTemplate']
            number = idx + 1  # Start from 1
            
            event_id_to_number[event_id] = number
            number_to_template[number] = template
            template_to_number[template] = number
        
        # Store mappings
        self.template_to_number = template_to_number
        self.number_to_template = number_to_template
        
        # Prepare for JSON serialization (convert int keys to strings)
        mapping_data = {
            'event_id_to_number': {str(k): v for k, v in event_id_to_number.items()},
            'number_to_template': {str(k): v for k, v in number_to_template.items()},
            'template_to_number': template_to_number,
            'total_templates': len(number_to_template)
        }
        
        # Save to JSON
        print(f"Saving mapping to: {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"Created mapping for {len(event_id_to_number)} templates")
        
        return event_id_to_number
    
    def load_mapping(self, mapping_json: str) -> None:
        """
        Load existing mapping from JSON.
        
        Args:
            mapping_json: Path to mapping JSON file
        """
        print(f"Loading template mapping from: {mapping_json}")
        
        with open(mapping_json, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Convert string keys back to integers for number_to_template
        self.number_to_template = {
            int(k): v for k, v in mapping_data['number_to_template'].items()
        }
        self.template_to_number = mapping_data['template_to_number']
        
        print(f"Loaded mapping for {len(self.number_to_template)} templates")
    
    def map_event_id_to_number(self, event_id: int, event_id_mapping: Dict[int, int]) -> int:
        """
        Map event ID to template number.
        
        Args:
            event_id: Original event ID from Drain
            event_id_mapping: Mapping from event_id to number
            
        Returns:
            Template number
        """
        return event_id_mapping.get(event_id, 0)
    
    def map_number_to_template(self, number: int) -> str:
        """
        Map template number back to template string.
        
        Args:
            number: Template number
            
        Returns:
            Template string
        """
        return self.number_to_template.get(number, "<UNKNOWN>")
    
    def get_template_text(self, numbers: list) -> str:
        """
        Convert a sequence of numbers to template text.
        
        Args:
            numbers: List of template numbers
            
        Returns:
            String with templates joined by separator
        """
        templates = [self.map_number_to_template(num) for num in numbers]
        return ' | '.join(templates)

