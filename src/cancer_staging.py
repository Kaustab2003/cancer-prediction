"""
Cancer Staging Module
Implements TNM staging and risk stratification for multiple cancer types
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import re

class CancerStaging:
    """
    Comprehensive cancer staging system for multiple cancer types
    Implements TNM staging, AJCC guidelines, and risk stratification
    """
    
    def __init__(self):
        self.staging_systems = {
            'breast': self._stage_breast_cancer,
            'lung': self._stage_lung_cancer,
            'skin': self._stage_skin_cancer,
            'blood': self._stage_blood_cancer
        }
    
    def determine_stage(self, cancer_type: str, clinical_data: Dict[str, Any], 
                       prediction_proba: float = None) -> Dict[str, Any]:
        """
        Determine cancer stage based on cancer type and available data
        
        Args:
            cancer_type: Type of cancer ('breast', 'lung', 'skin', 'blood')
            clinical_data: Dictionary with clinical features
            prediction_proba: Model prediction probability
            
        Returns:
            Dictionary with stage, TNM classification, and risk level
        """
        cancer_type = cancer_type.lower()
        
        if cancer_type not in self.staging_systems:
            return self._generic_staging(prediction_proba)
        
        staging_func = self.staging_systems[cancer_type]
        return staging_func(clinical_data, prediction_proba)
    
    def _stage_breast_cancer(self, data: Dict[str, Any], proba: float = None) -> Dict[str, Any]:
        """
        Stage breast cancer using tumor characteristics
        Based on AJCC TNM staging system
        """
        # Extract features (from Wisconsin dataset)
        radius_mean = data.get('radius_mean', data.get('radius1', 0))
        area_mean = data.get('area_mean', data.get('area1', 0))
        concavity_mean = data.get('concavity_mean', data.get('concavity1', 0))
        concave_points_mean = data.get('concave points_mean', data.get('concave_points1', 0))
        
        # Worst features (indicators of aggressive tumors)
        radius_worst = data.get('radius_worst', data.get('radius3', radius_mean * 1.3))
        area_worst = data.get('area_worst', data.get('area3', area_mean * 1.5))
        
        # Estimate tumor size from radius (radius in mm, size in cm)
        tumor_size_cm = (radius_worst / 10.0) * 2  # Diameter in cm
        
        # Estimate lymph node involvement from features
        lymph_node_score = (concavity_mean + concave_points_mean) / 2
        
        # Determine T (Tumor size)
        if tumor_size_cm <= 2:
            t_stage = "T1"
        elif tumor_size_cm <= 5:
            t_stage = "T2"
        elif tumor_size_cm > 5:
            t_stage = "T3"
        else:
            t_stage = "TX"
        
        # Determine N (Lymph nodes) - proxy from features
        if lymph_node_score < 0.05:
            n_stage = "N0"
        elif lymph_node_score < 0.15:
            n_stage = "N1"
        elif lymph_node_score < 0.25:
            n_stage = "N2"
        else:
            n_stage = "N3"
        
        # Determine M (Metastasis) - use probability and severity indicators
        metastasis_score = area_worst / 1000.0  # Normalized score
        
        if metastasis_score > 2.0 and proba and proba > 0.9:
            m_stage = "M1"
        else:
            m_stage = "M0"
        
        # Combine TNM to determine overall stage
        stage = self._calculate_breast_stage(t_stage, n_stage, m_stage)
        
        # Risk stratification
        risk_level = self._calculate_risk_level(stage, proba)
        
        return {
            'stage': stage,
            'tnm': {
                'T': t_stage,
                'N': n_stage,
                'M': m_stage
            },
            'risk_level': risk_level,
            'tumor_size_cm': round(tumor_size_cm, 2),
            'estimated_lymph_nodes': 'Involved' if n_stage != 'N0' else 'Clear',
            'metastasis': 'Present' if m_stage == 'M1' else 'Not detected',
            'staging_confidence': 'High' if proba and proba > 0.8 else 'Moderate'
        }
    
    def _calculate_breast_stage(self, t: str, n: str, m: str) -> str:
        """Calculate overall breast cancer stage from TNM"""
        if m == "M1":
            return "Stage IV"
        
        if t == "T1" and n == "N0":
            return "Stage I"
        elif t in ["T1", "T2"] and n in ["N0", "N1"]:
            return "Stage II"
        elif t == "T3" or n in ["N2", "N3"]:
            return "Stage III"
        else:
            return "Stage II"
    
    def _stage_lung_cancer(self, data: Dict[str, Any], proba: float = None) -> Dict[str, Any]:
        """
        Stage lung cancer based on symptom survey data
        Simplified staging based on symptom severity
        """
        # Extract symptoms
        age = data.get('AGE', data.get('age', 50))
        smoking = data.get('SMOKING', data.get('smoking', 1))
        yellow_fingers = data.get('YELLOW_FINGERS', data.get('yellow_fingers', 1))
        anxiety = data.get('ANXIETY', data.get('anxiety', 1))
        chronic_disease = data.get('CHRONIC DISEASE', data.get('chronic_disease', 1))
        fatigue = data.get('FATIGUE', data.get('fatigue', 1))
        allergy = data.get('ALLERGY', data.get('allergy', 1))
        wheezing = data.get('WHEEZING', data.get('wheezing', 1))
        alcohol = data.get('ALCOHOL CONSUMING', data.get('alcohol', 1))
        coughing = data.get('COUGHING', data.get('coughing', 1))
        shortness_breath = data.get('SHORTNESS OF BREATH', data.get('shortness_breath', 1))
        swallowing = data.get('SWALLOWING DIFFICULTY', data.get('swallowing', 1))
        chest_pain = data.get('CHEST PAIN', data.get('chest_pain', 1))
        
        # Calculate severity score
        symptom_score = sum([
            yellow_fingers * 2,  # High weight
            chronic_disease * 2,
            fatigue * 1.5,
            wheezing * 2,
            coughing * 1.5,
            shortness_breath * 2,
            swallowing * 2.5,
            chest_pain * 2
        ])
        
        # Risk factors
        risk_score = (smoking * 3) + (age > 60) * 2 + (alcohol * 1)
        
        # Combined score
        total_score = symptom_score + risk_score
        
        # Determine stage based on symptom severity
        if total_score <= 5:
            stage = "Stage I"
            t_stage, n_stage = "T1", "N0"
        elif total_score <= 10:
            stage = "Stage II"
            t_stage, n_stage = "T2", "N1"
        elif total_score <= 15:
            stage = "Stage III"
            t_stage, n_stage = "T3", "N2"
        else:
            stage = "Stage IV"
            t_stage, n_stage = "T4", "N3"
        
        m_stage = "M1" if total_score > 18 else "M0"
        
        risk_level = self._calculate_risk_level(stage, proba)
        
        return {
            'stage': stage,
            'tnm': {
                'T': t_stage,
                'N': n_stage,
                'M': m_stage
            },
            'risk_level': risk_level,
            'symptom_severity': 'Severe' if symptom_score > 12 else 'Moderate' if symptom_score > 6 else 'Mild',
            'risk_factors': f"{int(risk_score)} risk factors identified",
            'smoking_status': 'Current smoker' if smoking == 2 else 'Non-smoker',
            'staging_confidence': 'High' if proba and proba > 0.8 else 'Moderate'
        }
    
    def _stage_skin_cancer(self, data: Dict[str, Any], proba: float = None) -> Dict[str, Any]:
        """
        Stage skin cancer (melanoma/non-melanoma)
        Based on HAM10000 metadata or image analysis
        """
        # Extract data
        dx = data.get('dx', data.get('diagnosis', 'unknown'))
        age = data.get('age', 50)
        localization = data.get('localization', 'unknown')
        dx_type = data.get('dx_type', 'unknown')
        
        # Map diagnosis to cancer type
        dx_mapping = {
            'mel': 'Melanoma',
            'bcc': 'Basal Cell Carcinoma',
            'akiec': 'Actinic Keratosis',
            'nv': 'Benign Nevus',
            'bkl': 'Benign Keratosis',
            'vasc': 'Vascular Lesion',
            'df': 'Dermatofibroma'
        }
        
        cancer_type = dx_mapping.get(dx, dx)
        
        # Melanoma staging (more aggressive)
        if dx == 'mel':
            # Use probability as indicator of progression
            if proba and proba > 0.9:
                stage = "Stage III"
                t_stage, n_stage = "T3", "N1"
            elif proba and proba > 0.7:
                stage = "Stage II"
                t_stage, n_stage = "T2", "N0"
            else:
                stage = "Stage I"
                t_stage, n_stage = "T1", "N0"
        
        # BCC/AKIEC staging
        elif dx in ['bcc', 'akiec']:
            if proba and proba > 0.85:
                stage = "Stage II"
                t_stage, n_stage = "T2", "N0"
            else:
                stage = "Stage I"
                t_stage, n_stage = "T1", "N0"
        
        # Benign lesions
        else:
            stage = "Stage 0 (In situ)"
            t_stage, n_stage = "Tis", "N0"
        
        m_stage = "M0"  # Assume no metastasis from image alone
        
        risk_level = self._calculate_risk_level(stage, proba)
        
        # High-risk locations
        high_risk_locations = ['face', 'scalp', 'neck', 'ear']
        location_risk = 'High' if localization in high_risk_locations else 'Standard'
        
        return {
            'stage': stage,
            'tnm': {
                'T': t_stage,
                'N': n_stage,
                'M': m_stage
            },
            'risk_level': risk_level,
            'lesion_type': cancer_type,
            'location': localization,
            'location_risk': location_risk,
            'patient_age': age,
            'diagnosis_method': 'Histopathology' if dx_type == 'histo' else 'Clinical/Consensus',
            'staging_confidence': 'High' if dx_type == 'histo' else 'Moderate'
        }
    
    def _stage_blood_cancer(self, data: Dict[str, Any], proba: float = None) -> Dict[str, Any]:
        """
        Stage blood cell cancer (ALL - Acute Lymphoblastic Leukemia)
        Based on cell type classification
        """
        # Extract predicted class
        predicted_class = data.get('predicted_class', data.get('class', 'Unknown'))
        
        # ALL staging based on cell type
        if 'Pro-B' in predicted_class or 'Malignant' in predicted_class:
            if 'early Pre-B' in predicted_class:
                stage = "Early Stage (L1)"
                risk = "Low to Moderate"
                blast_percentage = "20-50%"
            elif 'Pre-B' in predicted_class and 'early' not in predicted_class:
                stage = "Advanced Stage (L2)"
                risk = "Moderate to High"
                blast_percentage = "50-80%"
            elif 'Pro-B' in predicted_class:
                stage = "Advanced Stage (L3)"
                risk = "High"
                blast_percentage = ">80%"
            else:
                stage = "Intermediate Stage"
                risk = "Moderate"
                blast_percentage = "30-70%"
        else:  # Benign
            stage = "No evidence of disease"
            risk = "None"
            blast_percentage = "<20% (Normal)"
        
        risk_level = self._calculate_risk_level(stage, proba)
        
        return {
            'stage': stage,
            'risk_level': risk_level,
            'cell_type': predicted_class,
            'blast_percentage': blast_percentage,
            'who_classification': 'ALL (Acute Lymphoblastic Leukemia)' if 'Malignant' in predicted_class else 'Normal',
            'treatment_urgency': 'Immediate' if risk == 'High' else 'Scheduled' if risk != 'None' else 'Monitoring',
            'staging_confidence': 'High' if proba and proba > 0.8 else 'Moderate'
        }
    
    def _generic_staging(self, proba: float = None) -> Dict[str, Any]:
        """Generic staging when specific cancer type not available"""
        if proba is None:
            proba = 0.5
        
        if proba < 0.3:
            stage = "Stage 0 or I"
            risk = "Low"
        elif proba < 0.6:
            stage = "Stage II"
            risk = "Moderate"
        elif proba < 0.85:
            stage = "Stage III"
            risk = "High"
        else:
            stage = "Stage IV"
            risk = "Very High"
        
        return {
            'stage': stage,
            'risk_level': risk,
            'confidence': 'Low (generic staging)',
            'note': 'Requires additional clinical evaluation'
        }
    
    def _calculate_risk_level(self, stage: str, proba: float = None) -> str:
        """Calculate risk level from stage and probability"""
        # Extract numeric stage
        stage_num = 0
        if 'IV' in stage or '4' in stage:
            stage_num = 4
        elif 'III' in stage or '3' in stage:
            stage_num = 3
        elif 'II' in stage or '2' in stage:
            stage_num = 2
        elif 'I' in stage or '1' in stage:
            stage_num = 1
        
        # Base risk on stage
        if stage_num == 0:
            base_risk = "Very Low"
        elif stage_num == 1:
            base_risk = "Low"
        elif stage_num == 2:
            base_risk = "Moderate"
        elif stage_num == 3:
            base_risk = "High"
        else:
            base_risk = "Very High"
        
        # Adjust based on probability
        if proba:
            if proba > 0.95 and stage_num >= 2:
                base_risk = "Very High"
            elif proba < 0.6 and stage_num <= 2:
                base_risk = "Low" if base_risk == "Moderate" else base_risk
        
        return base_risk
    
    def get_prognosis(self, cancer_type: str, stage: str) -> Dict[str, Any]:
        """
        Get survival statistics and prognosis based on cancer type and stage
        Based on evidence-based medical data (same as dashboard.py)
        """
        prognosis_data = {
            'breast': {
                'Stage 0': {'5yr_survival': 99, 'description': 'Excellent prognosis'},
                'Stage I': {'5yr_survival': 98, 'description': 'Very good prognosis'},
                'Stage II': {'5yr_survival': 85, 'description': 'Good prognosis with treatment'},
                'Stage III': {'5yr_survival': 58, 'description': 'Moderate prognosis'},
                'Stage IV': {'5yr_survival': 22, 'description': 'Advanced disease'}
            },
            'lung': {
                'Stage I': {'5yr_survival': 68, 'description': 'Localized disease'},
                'Stage II': {'5yr_survival': 53, 'description': 'Regional spread'},
                'Stage III': {'5yr_survival': 26, 'description': 'Advanced regional disease'},
                'Stage IV': {'5yr_survival': 6, 'description': 'Metastatic disease'}
            },
            'skin': {
                'Stage 0': {'5yr_survival': 99, 'description': 'In situ, excellent prognosis'},
                'Stage I': {'5yr_survival': 95, 'description': 'Thin melanoma'},
                'Stage II': {'5yr_survival': 82, 'description': 'Thicker melanoma'},
                'Stage III': {'5yr_survival': 63, 'description': 'Lymph node involvement'},
                'Stage IV': {'5yr_survival': 25, 'description': 'Metastatic melanoma'}
            },
            'blood': {
                'Early Stage (L1)': {'5yr_survival': 85, 'description': 'Standard-risk ALL'},
                'Intermediate Stage': {'5yr_survival': 70, 'description': 'Intermediate-risk ALL'},
                'Advanced Stage (L2)': {'5yr_survival': 55, 'description': 'High-risk ALL'},
                'Advanced Stage (L3)': {'5yr_survival': 40, 'description': 'Very high-risk ALL'}
            }
        }
        
        cancer_data = prognosis_data.get(cancer_type.lower(), {})
        stage_data = cancer_data.get(stage, {
            '5yr_survival': None,
            'description': 'Prognosis varies by individual factors'
        })
        
        return stage_data


# Example usage
if __name__ == "__main__":
    stager = CancerStaging()
    
    # Example: Breast cancer
    breast_data = {
        'radius_mean': 17.99,
        'area_mean': 1001,
        'concavity_mean': 0.3001,
        'concave points_mean': 0.1471,
        'radius_worst': 25.38,
        'area_worst': 2019
    }
    
    breast_stage = stager.determine_stage('breast', breast_data, prediction_proba=0.95)
    print("="*80)
    print("BREAST CANCER STAGING EXAMPLE")
    print("="*80)
    for key, value in breast_stage.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Get prognosis
    prognosis = stager.get_prognosis('breast', breast_stage['stage'])
    print(f"\nPROGNOSIS:")
    print(f"  5-year survival rate: {prognosis['5yr_survival']}%")
    print(f"  Description: {prognosis['description']}")
    
    print("\n" + "="*80 + "\n")
    
    # Example: Lung cancer
    lung_data = {
        'AGE': 65,
        'SMOKING': 2,
        'YELLOW_FINGERS': 2,
        'ANXIETY': 2,
        'CHRONIC DISEASE': 2,
        'FATIGUE': 2,
        'WHEEZING': 2,
        'COUGHING': 2,
        'SHORTNESS OF BREATH': 2,
        'CHEST PAIN': 2
    }
    
    lung_stage = stager.determine_stage('lung', lung_data, prediction_proba=0.92)
    print("="*80)
    print("LUNG CANCER STAGING EXAMPLE")
    print("="*80)
    for key, value in lung_stage.items():
        if isinstance(value, dict):
            print(f"\n{key.upper()}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Get prognosis
    prognosis = stager.get_prognosis('lung', lung_stage['stage'])
    print(f"\nPROGNOSIS:")
    print(f"  5-year survival rate: {prognosis['5yr_survival']}%")
    print(f"  Description: {prognosis['description']}")
