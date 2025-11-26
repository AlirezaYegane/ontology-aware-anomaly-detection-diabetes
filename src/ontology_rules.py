"""
Ontology Rules Module

Implementation of domain-specific ontological constraints and rules
for validating anomalies in the diabetes dataset.
"""

import pandas as pd
import numpy as np


class OntologyRuleEngine:
    """
    Rule engine for validating anomalies using medical domain knowledge.
    """
    
    def __init__(self):
        self.rules = []
        self.rule_violations = {}
        
    def add_rule(self, name, condition_func, description=""):
        """
        Add a validation rule.
        
        Parameters
        ----------
        name : str
            Unique rule identifier
        condition_func : callable
            Function that takes a DataFrame row and returns True if valid
        description : str
            Human-readable rule description
        """
        self.rules.append({
            'name': name,
            'condition': condition_func,
            'description': description
        })
        
    def validate(self, df):
        """
        Validate all rules against the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate
            
        Returns
        -------
        pd.DataFrame
            Original dataframe with rule violation flags
        """
        result_df = df.copy()
        
        for rule in self.rules:
            rule_name = rule['name']
            condition = rule['condition']
            
            # Apply rule to each row
            violations = ~df.apply(condition, axis=1)
            result_df[f'violates_{rule_name}'] = violations
            
            n_violations = violations.sum()
            self.rule_violations[rule_name] = n_violations
            
            print(f"Rule '{rule_name}': {n_violations} violations "
                  f"({n_violations/len(df)*100:.2f}%)")
        
        # Add overall violation flag
        violation_cols = [col for col in result_df.columns if col.startswith('violates_')]
        result_df['has_rule_violation'] = result_df[violation_cols].any(axis=1)
        
        return result_df
    
    def get_violation_summary(self):
        """Get summary of rule violations."""
        return pd.DataFrame.from_dict(
            self.rule_violations, 
            orient='index', 
            columns=['violations']
        )


def create_diabetes_rules():
    """
    Create ontology rules specific to the diabetes dataset.
    
    Returns
    -------
    OntologyRuleEngine
        Rule engine with diabetes-specific rules
    """
    engine = OntologyRuleEngine()
    
    # Rule 1: Age-medication compatibility
    def age_medication_rule(row):
        """Children (<18) should not have certain adult medications."""
        # This is a simplified example - adjust based on actual column names
        if 'age' in row.index and 'metformin' in row.index:
            age_bracket = row.get('age', '')
            metformin = row.get('metformin', 'No')
            
            if 'child' in str(age_bracket).lower() and metformin != 'No':
                return False  # Violation
        return True
    
    engine.add_rule(
        'age_medication',
        age_medication_rule,
        "Children should not receive certain adult diabetes medications"
    )
    
    # Rule 2: Diagnosis-medication consistency
    def diagnosis_medication_rule(row):
        """Diabetes medications should align with diabetes diagnosis."""
        # Example: if diabetic medication but no diabetes diagnosis
        diabetes_meds = ['metformin', 'insulin', 'glyburide']
        
        has_diabetes_med = False
        for med in diabetes_meds:
            if med in row.index and row.get(med, 'No') != 'No':
                has_diabetes_med = True
                break
        
        # Check if diabetes diagnosis exists
        if has_diabetes_med:
            # Simplified - check if any diagnosis code relates to diabetes
            # In practice, you'd check ICD-9 codes starting with 250
            for diag_col in ['diag_1', 'diag_2', 'diag_3']:
                if diag_col in row.index:
                    diag_code = str(row.get(diag_col, ''))
                    if diag_code.startswith('250'):  # Diabetes ICD-9 codes
                        return True
            return False  # Has medication but no diabetes diagnosis
        
        return True
    
    engine.add_rule(
        'diagnosis_medication',
        diagnosis_medication_rule,
        "Diabetes medications should correspond to diabetes diagnosis"
    )
    
    # Rule 3: Length of stay reasonableness
    def length_of_stay_rule(row):
        """Length of stay should be within reasonable bounds."""
        if 'time_in_hospital' in row.index:
            los = row.get('time_in_hospital', 0)
            if los > 14:  # More than 2 weeks is unusual
                return False
        return True
    
    engine.add_rule(
        'length_of_stay',
        length_of_stay_rule,
        "Length of hospital stay should be within reasonable range"
    )
    
    # Rule 4: Number of procedures
    def procedures_rule(row):
        """Excessive number of procedures may indicate data quality issues."""
        if 'num_procedures' in row.index:
            n_procedures = row.get('num_procedures', 0)
            if n_procedures > 6:  # Threshold for unusual number
                return False
        return True
    
    engine.add_rule(
        'excess_procedures',
        procedures_rule,
        "Number of procedures should not be excessive"
    )
    
    return engine


def validate_with_ontology(df, anomaly_col='is_anomaly'):
    """
    Validate anomalies using ontological rules.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with anomaly predictions
    anomaly_col : str
        Column name containing anomaly flags
        
    Returns
    -------
    pd.DataFrame
        Dataset with ontology validation results
    dict
        Analysis of anomalies vs rule violations
    """
    print("Applying ontology rules...")
    
    # Create and apply rules
    engine = create_diabetes_rules()
    validated_df = engine.validate(df)
    
    # Analyze overlap between ML anomalies and rule violations
    if anomaly_col in validated_df.columns:
        ml_anomalies = validated_df[anomaly_col]
        rule_violations = validated_df['has_rule_violation']
        
        both = (ml_anomalies & rule_violations).sum()
        only_ml = (ml_anomalies & ~rule_violations).sum()
        only_rules = (~ml_anomalies & rule_violations).sum()
        
        analysis = {
            'total_ml_anomalies': ml_anomalies.sum(),
            'total_rule_violations': rule_violations.sum(),
            'confirmed_anomalies': both,  # Both ML and rules agree
            'ml_only_anomalies': only_ml,
            'rule_only_violations': only_rules,
            'rule_violation_summary': engine.get_violation_summary()
        }
        
        print(f"\nValidation Summary:")
        print(f"  ML detected: {ml_anomalies.sum()} anomalies")
        print(f"  Rules found: {rule_violations.sum()} violations")
        print(f"  Confirmed (both): {both}")
        print(f"  ML only: {only_ml}")
        print(f"  Rules only: {only_rules}")
        
        return validated_df, analysis
    
    return validated_df, {'rule_violation_summary': engine.get_violation_summary()}
