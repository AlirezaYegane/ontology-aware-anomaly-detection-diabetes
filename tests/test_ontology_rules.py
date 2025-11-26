"""
Unit Tests for Ontology-Inspired Rule Layer
===========================================

This script tests the logic of the ontology rule layer to ensure
clinical rules are applied correctly.

Usage:
    python -m unittest tests/test_ontology_rules.py
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ontology_rule_layer import compute_ontology_penalty, evaluate_anomaly_scores

class TestOntologyRules(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        pass

    def test_rule_1_high_risk(self):
        """
        Test Rule 1: Poor glycemic control + no med change + on diabetes meds
        Expected Penalty: 0.9
        """
        row = pd.Series({
            'A1Cresult': '>8',
            'change': 'No',
            'diabetesMed': 'Yes',
            'max_glu_serum': 'None',
            'num_lab_procedures': 50,
            'num_medications': 10,
            'time_in_hospital': 5
        })
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.9, "Rule 1 should return 0.9 penalty")

        # Test variant with >7
        row['A1Cresult'] = '>7'
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.9, "Rule 1 (>7) should return 0.9 penalty")

    def test_rule_2_high_risk(self):
        """
        Test Rule 2: Very high glucose + insufficient lab monitoring
        Expected Penalty: 0.85
        """
        row = pd.Series({
            'A1Cresult': 'None',
            'change': 'Ch',
            'diabetesMed': 'Yes',
            'max_glu_serum': '>300',
            'num_lab_procedures': 20,  # < 40
            'num_medications': 10,
            'time_in_hospital': 5
        })
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.85, "Rule 2 should return 0.85 penalty")

        # Test variant with >200
        row['max_glu_serum'] = '>200'
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.85, "Rule 2 (>200) should return 0.85 penalty")

    def test_rule_3_medium_risk(self):
        """
        Test Rule 3: High medication burden + short hospital stay
        Expected Penalty: 0.6
        """
        row = pd.Series({
            'A1Cresult': 'None',
            'change': 'Ch',
            'diabetesMed': 'Yes',
            'max_glu_serum': 'None',
            'num_lab_procedures': 50,
            'num_medications': 25,  # > 20
            'time_in_hospital': 2   # < 3
        })
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.6, "Rule 3 should return 0.6 penalty")

    def test_low_risk_default(self):
        """
        Test Default: No rules met
        Expected Penalty: 0.1
        """
        row = pd.Series({
            'A1Cresult': 'Norm',
            'change': 'No',
            'diabetesMed': 'No',
            'max_glu_serum': 'Norm',
            'num_lab_procedures': 50,
            'num_medications': 10,
            'time_in_hospital': 5
        })
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.1, "Default should be 0.1")

    def test_rule_priority(self):
        """
        Test that highest penalty takes precedence.
        Trigger Rule 1 (0.9) and Rule 3 (0.6) simultaneously.
        """
        row = pd.Series({
            'A1Cresult': '>8',       # Rule 1 trigger
            'change': 'No',          # Rule 1 trigger
            'diabetesMed': 'Yes',    # Rule 1 trigger
            'max_glu_serum': 'None',
            'num_lab_procedures': 50,
            'num_medications': 25,   # Rule 3 trigger
            'time_in_hospital': 2    # Rule 3 trigger
        })
        penalty = compute_ontology_penalty(row)
        self.assertEqual(penalty, 0.9, "Should take max penalty (0.9 over 0.6)")

    def test_evaluate_anomaly_scores(self):
        """Test evaluation metrics calculation"""
        y_true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        metrics = evaluate_anomaly_scores(y_true, scores, "Test Score")
        
        self.assertIn('roc_auc', metrics)
        self.assertIn('pr_auc', metrics)
        self.assertEqual(metrics['score_name'], "Test Score")
        
        # Basic sanity check on values
        self.assertTrue(0 <= metrics['roc_auc'] <= 1)
        self.assertTrue(0 <= metrics['pr_auc'] <= 1)

if __name__ == '__main__':
    unittest.main()
