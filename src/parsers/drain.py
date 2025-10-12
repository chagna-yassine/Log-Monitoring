"""
Drain Log Parser Implementation

Based on the paper:
"Drain: An Online Log Parsing Approach with Fixed Depth Tree"
by Pinjia He et al.

This implementation parses raw log entries into structured format and extracts event templates.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class LogCluster:
    """Represents a cluster of similar log messages."""
    
    def __init__(self, log_template_tokens: List[str], cluster_id: int):
        self.log_template_tokens = log_template_tokens
        self.cluster_id = cluster_id
        self.log_ids = []
        
    def get_template(self) -> str:
        """Get the log template as a string."""
        return ' '.join(self.log_template_tokens)


class Node:
    """Node in the parse tree."""
    
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.children = {}  # key: token or token length, value: Node
        self.clusters = []  # List of LogCluster objects


class Drain:
    """
    Drain log parser for extracting event templates from logs.
    
    Args:
        depth: Depth of the parse tree (default: 4)
        st: Similarity threshold for matching templates (default: 0.4)
        rex: List of regular expressions for preprocessing
    """
    
    def __init__(
        self,
        depth: int = 4,
        st: float = 0.4,
        rex: Optional[List[str]] = None
    ):
        self.depth = depth
        self.st = st
        self.rex = rex or []
        
        # Compile regex patterns
        self.rex_patterns = [re.compile(pattern) for pattern in self.rex]
        
        # Parse tree root
        self.root = Node(depth=0)
        
        # Store all clusters by ID
        self.clusters = {}
        self.cluster_counter = 0
        
    def preprocess(self, content: str) -> str:
        """
        Preprocess log content by masking variables with regex.
        
        Args:
            content: Raw log content
            
        Returns:
            Preprocessed content with variables masked
        """
        for pattern in self.rex_patterns:
            content = pattern.sub('<*>', content)
        return content
    
    def tokenize(self, content: str) -> List[str]:
        """
        Tokenize log content into words.
        
        Args:
            content: Log content
            
        Returns:
            List of tokens
        """
        return content.split()
    
    def fast_match(self, cluster_list: List[LogCluster], tokens: List[str]) -> Optional[LogCluster]:
        """
        Fast matching against existing clusters.
        
        Args:
            cluster_list: List of candidate clusters
            tokens: Tokenized log content
            
        Returns:
            Matched cluster or None
        """
        for cluster in cluster_list:
            template_tokens = cluster.log_template_tokens
            
            if len(template_tokens) != len(tokens):
                continue
                
            # Calculate similarity
            similarities = sum(
                1 for token, template_token in zip(tokens, template_tokens)
                if token == template_token
            )
            
            similarity = similarities / len(tokens)
            
            if similarity >= self.st:
                return cluster
                
        return None
    
    def get_template(self, cluster_tokens: List[str], tokens: List[str]) -> List[str]:
        """
        Update template by comparing with new tokens.
        
        Args:
            cluster_tokens: Existing cluster template tokens
            tokens: New log tokens
            
        Returns:
            Updated template tokens
        """
        new_template = []
        
        for cluster_token, token in zip(cluster_tokens, tokens):
            if cluster_token == token:
                new_template.append(token)
            else:
                new_template.append('<*>')
                
        return new_template
    
    def add_log_to_cluster(self, cluster: LogCluster, log_id: int, tokens: List[str]) -> None:
        """
        Add a log entry to an existing cluster and update template.
        
        Args:
            cluster: Target cluster
            log_id: Log entry ID
            tokens: Log tokens
        """
        cluster.log_ids.append(log_id)
        
        # Update template
        if cluster.log_template_tokens != tokens:
            cluster.log_template_tokens = self.get_template(
                cluster.log_template_tokens, tokens
            )
    
    def create_cluster(self, tokens: List[str], log_id: int, node: Node) -> LogCluster:
        """
        Create a new cluster for the log.
        
        Args:
            tokens: Log tokens
            log_id: Log entry ID
            node: Parent node
            
        Returns:
            New cluster
        """
        self.cluster_counter += 1
        cluster = LogCluster(
            log_template_tokens=tokens.copy(),
            cluster_id=self.cluster_counter
        )
        cluster.log_ids.append(log_id)
        
        # Add to node and global storage
        node.clusters.append(cluster)
        self.clusters[self.cluster_counter] = cluster
        
        return cluster
    
    def tree_search(self, node: Node, tokens: List[str], depth: int) -> Node:
        """
        Search the parse tree to find the appropriate leaf node.
        
        Args:
            node: Current node
            tokens: Log tokens
            depth: Current depth
            
        Returns:
            Leaf node
        """
        if depth >= self.depth:
            return node
        
        # First layer: group by log length
        if depth == 0:
            log_length = len(tokens)
            if log_length not in node.children:
                node.children[log_length] = Node(depth=depth + 1)
            return self.tree_search(node.children[log_length], tokens, depth + 1)
        
        # Subsequent layers: group by token at position
        token_idx = depth - 1
        
        if token_idx >= len(tokens):
            # Use wildcard if token doesn't exist at this position
            if '<*>' not in node.children:
                node.children['<*>'] = Node(depth=depth + 1)
            return self.tree_search(node.children['<*>'], tokens, depth + 1)
        
        token = tokens[token_idx]
        
        # Use wildcard for numbers or masked tokens
        if token == '<*>' or self._is_number(token):
            if '<*>' not in node.children:
                node.children['<*>'] = Node(depth=depth + 1)
            return self.tree_search(node.children['<*>'], tokens, depth + 1)
        
        # Use actual token
        if token not in node.children:
            node.children[token] = Node(depth=depth + 1)
        return self.tree_search(node.children[token], tokens, depth + 1)
    
    def _is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
    
    def parse(self, log_id: int, content: str) -> Tuple[int, str]:
        """
        Parse a single log entry.
        
        Args:
            log_id: Unique log identifier
            content: Raw log content
            
        Returns:
            Tuple of (event_id, event_template)
        """
        # Preprocess and tokenize
        preprocessed = self.preprocess(content)
        tokens = self.tokenize(preprocessed)
        
        if not tokens:
            # Empty log
            return -1, ""
        
        # Search parse tree
        leaf_node = self.tree_search(self.root, tokens, depth=0)
        
        # Fast match against clusters in leaf node
        matched_cluster = self.fast_match(leaf_node.clusters, tokens)
        
        if matched_cluster:
            # Add to existing cluster
            self.add_log_to_cluster(matched_cluster, log_id, tokens)
            return matched_cluster.cluster_id, matched_cluster.get_template()
        else:
            # Create new cluster
            new_cluster = self.create_cluster(tokens, log_id, leaf_node)
            return new_cluster.cluster_id, new_cluster.get_template()
    
    def get_templates(self) -> Dict[int, str]:
        """
        Get all extracted templates.
        
        Returns:
            Dictionary mapping event_id to template string
        """
        return {
            cluster_id: cluster.get_template()
            for cluster_id, cluster in self.clusters.items()
        }
    
    def get_cluster_stats(self) -> Dict[int, int]:
        """
        Get statistics about cluster sizes.
        
        Returns:
            Dictionary mapping event_id to number of logs in cluster
        """
        return {
            cluster_id: len(cluster.log_ids)
            for cluster_id, cluster in self.clusters.items()
        }

