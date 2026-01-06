"""
Unified Multi-Cancer Prediction Pipeline
Integrates all cancer detection models, staging, and prevention recommendations
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import joblib
from datetime import datetime
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from cancer_staging import CancerStaging
from prevention_module import PreventionAdvisor

# Try importing PyTorch for image models
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA version: {torch.version.cuda}")
    else:
        print("[INFO] Using CPU for inference")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("Warning: PyTorch not available. Image-based predictions disabled.")


class MultiCancerPredictor:
    """
    Unified cancer prediction system handling multiple cancer types
    Supports both tabular clinical data and medical images
    """
    
    def __init__(self, models_dir: str = '.'):
        """
        Initialize multi-cancer predictor with all models
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        self.stager = CancerStaging()
        self.advisor = PreventionAdvisor()
        
        # Load all available models
        self._load_models()
    
    def _load_models(self):
        """Load all trained cancer detection models"""
        print("Loading cancer detection models...")
        
        # Load tabular models (breast, lung, synthetic)
        tabular_models = {
            'breast': ['breast_cancer_model.pkl', 'breast_metadata.json'],
            'lung': ['lung_cancer_model.pkl', 'lung_metadata.json'],
            'synthetic': ['advanced_cancer_model.pkl', 'model_metadata.json']
        }
        
        for cancer_type, (model_file, metadata_file) in tabular_models.items():
            model_path = self.models_dir / model_file
            meta_path = self.models_dir / metadata_file
            
            if model_path.exists():
                try:
                    self.models[cancer_type] = joblib.load(model_path)
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            self.metadata[cancer_type] = json.load(f)
                    print(f"✓ Loaded {cancer_type} cancer model")
                except Exception as e:
                    print(f"✗ Failed to load {cancer_type} model: {e}")
        
        # Load image-based models (blood, skin) - PyTorch models
        if TORCH_AVAILABLE:
            image_models = {
                'blood': ['blood_cancer_efficientnet_pytorch_best.pth', 'blood_cancer_efficientnet_pytorch_metadata.json'],
                'skin': ['skin_cancer_efficientnet_pytorch_best.pth', 'skin_cancer_efficientnet_pytorch_metadata.json']
            }
            
            for cancer_type, (model_file, metadata_file) in image_models.items():
                model_path = self.models_dir / model_file
                meta_path = self.models_dir / metadata_file
                
                if model_path.exists():
                    try:
                        # Load model architecture based on cancer type
                        if cancer_type == 'blood':
                            model = self._create_blood_model()
                        elif cancer_type == 'skin':
                            model = self._create_skin_model()
                        
                        # Load weights
                        checkpoint = torch.load(model_path, map_location=DEVICE)
                        state_dict = checkpoint['model_state_dict']
                        
                        # Remove 'backbone.' prefix if present
                        if any(k.startswith('backbone.') for k in state_dict.keys()):
                            state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                        
                        model.load_state_dict(state_dict)
                        model = model.to(DEVICE)
                        model.eval()
                        
                        self.models[cancer_type] = model
                        
                        if meta_path.exists():
                            with open(meta_path, 'r') as f:
                                self.metadata[cancer_type] = json.load(f)
                        print(f"✓ Loaded {cancer_type} cancer image model (PyTorch)")
                    except Exception as e:
                        print(f"✗ Failed to load {cancer_type} image model: {e}")
        
        print(f"\nTotal models loaded: {len(self.models)}")
    
    def _create_blood_model(self):
        """Create EfficientNetB3 model for blood cancer (4 classes)"""
        model = models.efficientnet_b3(pretrained=False)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 4)
        )
        return model
    
    def _create_skin_model(self):
        """Create EfficientNetB4 model for skin cancer (7 classes)"""
        model = models.efficientnet_b4(pretrained=False)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 7)
        )
        return model
    
    def predict(
        self,
        data: Union[Dict, np.ndarray, str, Image.Image],
        cancer_type: Optional[str] = None,
        include_staging: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Unified prediction interface for all cancer types
        
        Args:
            data: Input data (clinical features dict, image path, or PIL Image)
            cancer_type: Specific cancer type or None for auto-detection
            include_staging: Whether to include staging information
            include_recommendations: Whether to include prevention recommendations
            
        Returns:
            Comprehensive prediction results with staging and recommendations
        """
        # Determine input type
        is_image = isinstance(data, (str, Path, Image.Image, np.ndarray))
        
        if cancer_type:
            # Use specified cancer type
            return self._predict_specific(
                data, cancer_type, is_image, include_staging, include_recommendations
            )
        else:
            # Auto-detect cancer type or use multi-cancer approach
            return self._predict_auto(
                data, is_image, include_staging, include_recommendations
            )
    
    def _predict_specific(
        self,
        data: Any,
        cancer_type: str,
        is_image: bool,
        include_staging: bool,
        include_recommendations: bool
    ) -> Dict[str, Any]:
        """Make prediction for specific cancer type"""
        cancer_type = cancer_type.lower()
        
        if cancer_type not in self.models:
            return {
                'success': False,
                'error': f"Model for {cancer_type} cancer not available",
                'available_models': list(self.models.keys())
            }
        
        try:
            if is_image:
                prediction_result = self._predict_image(data, cancer_type)
            else:
                prediction_result = self._predict_tabular(data, cancer_type)
            
            # Add staging if requested
            if include_staging and prediction_result['success']:
                staging_result = self._add_staging(
                    cancer_type,
                    prediction_result,
                    data if not is_image else {}
                )
                prediction_result['staging'] = staging_result
            
            # Add recommendations if requested
            if include_recommendations and prediction_result['success']:
                recommendations = self._add_recommendations(
                    cancer_type,
                    prediction_result.get('staging', {}),
                    data if not is_image else {}
                )
                prediction_result['recommendations'] = recommendations
            
            return prediction_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}",
                'cancer_type': cancer_type
            }
    
    def _predict_tabular(self, data: Dict, cancer_type: str) -> Dict[str, Any]:
        """Predict using tabular clinical data"""
        model = self.models[cancer_type]
        metadata = self.metadata.get(cancer_type, {})
        
        # Prepare features
        if isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame([data])
            
            # Get expected features from metadata
            expected_features = metadata.get('feature_names', df.columns.tolist())
            
            # Ensure all expected features present
            for feature in expected_features:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Select and order features
            X = df[expected_features].values
        else:
            X = np.array(data).reshape(1, -1)
        
        # Make prediction
        try:
            prediction_proba = model.predict_proba(X)[0]
            prediction_class = model.predict(X)[0]
            
            # Get class names
            class_names = metadata.get('class_names', ['Benign', 'Malignant'])
            
            result = {
                'success': True,
                'cancer_type': cancer_type,
                'prediction': class_names[prediction_class],
                'confidence': float(prediction_proba[prediction_class]),
                'probabilities': {
                    class_names[i]: float(prediction_proba[i])
                    for i in range(len(class_names))
                },
                'is_malignant': prediction_class == 1 if len(class_names) == 2 else class_names[prediction_class] != 'Benign',
                'model_type': 'tabular',
                'model_info': {
                    'name': metadata.get('model_name', 'Unknown'),
                    'accuracy': metadata.get('test_accuracy', None),
                    'auc': metadata.get('test_auc', None)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}",
                'cancer_type': cancer_type
            }
    
    def _predict_image(self, data: Any, cancer_type: str) -> Dict[str, Any]:
        """Predict using medical image with PyTorch"""
        if not TORCH_AVAILABLE:
            return {
                'success': False,
                'error': "PyTorch not available for image predictions"
            }
        
        model = self.models[cancer_type]
        metadata = self.metadata.get(cancer_type, {})
        
        # Load and preprocess image
        try:
            if isinstance(data, str) or isinstance(data, Path):
                img = Image.open(data).convert('RGB')
            elif isinstance(data, Image.Image):
                img = data.convert('RGB')
            elif isinstance(data, np.ndarray):
                img = Image.fromarray(data).convert('RGB')
            else:
                return {
                    'success': False,
                    'error': "Invalid image data format"
                }
            
            # Get input size
            input_shape = metadata.get('input_shape', [3, 224, 224])
            target_size = (input_shape[1], input_shape[2])  # (H, W)
            
            # PyTorch preprocessing
            transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                prediction_class = torch.argmax(probabilities).item()
                prediction_proba = probabilities.cpu().numpy()
            
            # Get class names
            class_names = metadata.get('class_names', [])
            if not class_names:
                class_to_idx = metadata.get('class_to_idx', {})
                if class_to_idx:
                    class_names = [name for name, idx in sorted(class_to_idx.items(), key=lambda x: x[1])]
                else:
                    class_indices = metadata.get('class_indices', {})
                    class_names = [name for name, idx in sorted(class_indices.items(), key=lambda x: x[1])]
            
            # Determine if malignant
            predicted_label = class_names[prediction_class] if class_names else f"Class {prediction_class}"
            is_malignant = 'Malignant' in predicted_label or 'melanoma' in predicted_label.lower()
            
            result = {
                'success': True,
                'cancer_type': cancer_type,
                'prediction': predicted_label,
                'confidence': float(prediction_proba[prediction_class]),
                'probabilities': {
                    (class_names[i] if class_names else f"Class {i}"): float(prediction_proba[i])
                    for i in range(len(prediction_proba))
                },
                'is_malignant': is_malignant,
                'model_type': 'image',
                'model_info': {
                    'name': metadata.get('model_name', 'Unknown'),
                    'architecture': metadata.get('architecture', 'EfficientNet (PyTorch)'),
                    'accuracy': metadata.get('test_accuracy', None),
                    'auc': metadata.get('test_auc', None),
                    'device': str(DEVICE)
                },
                'image_info': {
                    'original_size': img.size,
                    'processed_size': target_size
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Image prediction error: {str(e)}",
                'cancer_type': cancer_type
            }
    
    def _predict_auto(
        self,
        data: Any,
        is_image: bool,
        include_staging: bool,
        include_recommendations: bool
    ) -> Dict[str, Any]:
        """Auto-detect cancer type and make predictions"""
        # For images, try image-based models
        if is_image:
            image_models = [ct for ct in ['blood', 'skin'] if ct in self.models]
            if not image_models:
                return {
                    'success': False,
                    'error': "No image-based models available"
                }
            
            # For now, use first available image model
            # In production, could use ensemble or router model
            cancer_type = image_models[0]
            return self._predict_specific(
                data, cancer_type, is_image, include_staging, include_recommendations
            )
        
        # For tabular data, try to detect which features are present
        else:
            if isinstance(data, dict):
                features = set(data.keys())
                
                # Breast cancer features (Wisconsin dataset)
                breast_features = {'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'}
                
                # Lung cancer features (symptom survey)
                lung_features = {'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'COUGHING'}
                
                if features & breast_features:
                    cancer_type = 'breast'
                elif features & lung_features:
                    cancer_type = 'lung'
                else:
                    cancer_type = 'synthetic'  # Default to general model
                
                return self._predict_specific(
                    data, cancer_type, is_image, include_staging, include_recommendations
                )
            
            return {
                'success': False,
                'error': "Cannot auto-detect cancer type from data format"
            }
    
    def _add_staging(
        self,
        cancer_type: str,
        prediction_result: Dict,
        clinical_data: Dict
    ) -> Dict[str, Any]:
        """Add staging information to prediction result"""
        # Prepare data for staging
        staging_data = clinical_data.copy() if clinical_data else {}
        
        # Add predicted class if available
        if 'prediction' in prediction_result:
            staging_data['predicted_class'] = prediction_result['prediction']
        
        # Get staging
        staging_result = self.stager.determine_stage(
            cancer_type=cancer_type,
            clinical_data=staging_data,
            prediction_proba=prediction_result.get('confidence', 0.5)
        )
        
        # Add prognosis
        if 'stage' in staging_result:
            prognosis = self.stager.get_prognosis(cancer_type, staging_result['stage'])
            staging_result['prognosis'] = prognosis
        
        return staging_result
    
    def _add_recommendations(
        self,
        cancer_type: str,
        staging_result: Dict,
        clinical_data: Dict
    ) -> Dict[str, Any]:
        """Add prevention and treatment recommendations"""
        stage = staging_result.get('stage', 'Unknown')
        risk_level = staging_result.get('risk_level', 'Moderate')
        
        recommendations = self.advisor.get_comprehensive_recommendations(
            cancer_type=cancer_type,
            stage=stage,
            risk_level=risk_level,
            patient_data=clinical_data
        )
        
        return recommendations
    
    def predict_batch(
        self,
        data_list: list,
        cancer_type: Optional[str] = None,
        include_staging: bool = True,
        include_recommendations: bool = False  # Default False for batch to reduce size
    ) -> list:
        """
        Make predictions for multiple samples
        
        Args:
            data_list: List of input data samples
            cancer_type: Cancer type (optional)
            include_staging: Include staging info
            include_recommendations: Include recommendations (expensive for large batches)
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, data in enumerate(data_list):
            result = self.predict(
                data=data,
                cancer_type=cancer_type,
                include_staging=include_staging,
                include_recommendations=include_recommendations
            )
            result['batch_index'] = i
            results.append(result)
        
        return results
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        info = {
            'total_models': len(self.models),
            'models': {}
        }
        
        for cancer_type, model in self.models.items():
            metadata = self.metadata.get(cancer_type, {})
            is_pytorch_model = isinstance(model, nn.Module) if TORCH_AVAILABLE else False
            info['models'][cancer_type] = {
                'type': 'image (PyTorch)' if is_pytorch_model else 'tabular',
                'name': metadata.get('model_name', 'Unknown'),
                'accuracy': metadata.get('test_accuracy'),
                'num_classes': metadata.get('num_classes'),
                'class_names': metadata.get('class_names', []),
                'device': str(DEVICE) if is_pytorch_model else 'CPU'
            }
        
        return info
    
    def generate_report(
        self,
        prediction_result: Dict,
        output_format: str = 'json',
        output_file: Optional[str] = None
    ) -> Union[Dict, str]:
        """
        Generate comprehensive patient report
        
        Args:
            prediction_result: Prediction result from predict()
            output_format: 'json', 'text', or 'html'
            output_file: Optional file path to save report
            
        Returns:
            Report in specified format
        """
        if output_format == 'json':
            report = prediction_result
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
            return report
        
        elif output_format == 'text':
            report_lines = []
            report_lines.append("="*80)
            report_lines.append("COMPREHENSIVE CANCER DETECTION REPORT")
            report_lines.append("="*80)
            report_lines.append(f"\nGenerated: {prediction_result.get('timestamp', datetime.now().isoformat())}")
            report_lines.append(f"Cancer Type: {prediction_result.get('cancer_type', 'Unknown').title()}")
            report_lines.append(f"\n{'='*80}")
            report_lines.append("PREDICTION RESULTS")
            report_lines.append("="*80)
            report_lines.append(f"Diagnosis: {prediction_result.get('prediction', 'Unknown')}")
            report_lines.append(f"Confidence: {prediction_result.get('confidence', 0)*100:.2f}%")
            report_lines.append(f"Malignancy: {'Yes' if prediction_result.get('is_malignant') else 'No'}")
            
            if 'staging' in prediction_result:
                staging = prediction_result['staging']
                report_lines.append(f"\n{'='*80}")
                report_lines.append("STAGING INFORMATION")
                report_lines.append("="*80)
                report_lines.append(f"Stage: {staging.get('stage', 'Unknown')}")
                report_lines.append(f"Risk Level: {staging.get('risk_level', 'Unknown')}")
                
                if 'tnm' in staging:
                    tnm = staging['tnm']
                    report_lines.append(f"TNM Classification: T{tnm.get('T', '?')}, N{tnm.get('N', '?')}, M{tnm.get('M', '?')}")
                
                if 'prognosis' in staging:
                    prognosis = staging['prognosis']
                    if '5yr_survival' in prognosis:
                        report_lines.append(f"5-Year Survival: {prognosis['5yr_survival']}%")
                    report_lines.append(f"Description: {prognosis.get('description', '')}")
            
            if 'recommendations' in prediction_result:
                rec = prediction_result['recommendations']
                report_lines.append(f"\n{'='*80}")
                report_lines.append("IMMEDIATE ACTIONS REQUIRED")
                report_lines.append("="*80)
                for action in rec.get('immediate_actions', [])[:5]:
                    report_lines.append(f"• {action}")
                
                report_lines.append(f"\n{'='*80}")
                report_lines.append("TOP LIFESTYLE RECOMMENDATIONS")
                report_lines.append("="*80)
                lifestyle = rec.get('lifestyle_modifications', {})
                for category, items in list(lifestyle.items())[:2]:
                    report_lines.append(f"\n{category.replace('_', ' ').title()}:")
                    for item in items[:3]:
                        report_lines.append(f"  - {item}")
            
            report = '\n'.join(report_lines)
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report)
            
            return report
        
        else:
            return {'error': f"Unsupported output format: {output_format}"}


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("MULTI-CANCER PREDICTION SYSTEM")
    print("="*80)
    
    # Initialize predictor
    predictor = MultiCancerPredictor()
    
    # Show available models
    models_info = predictor.get_available_models()
    print(f"\n✓ Loaded {models_info['total_models']} cancer detection models:")
    for cancer_type, info in models_info['models'].items():
        print(f"  - {cancer_type.title()}: {info['name']} ({info['type']})")
        if info['accuracy']:
            print(f"    Accuracy: {info['accuracy']*100:.2f}%")
    
    # Example 1: Breast cancer prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: Breast Cancer Prediction (Tabular Data)")
    print("="*80)
    
    breast_data = {
        'radius_mean': 17.99,
        'texture_mean': 10.38,
        'perimeter_mean': 122.8,
        'area_mean': 1001,
        'smoothness_mean': 0.1184,
        'compactness_mean': 0.2776,
        'concavity_mean': 0.3001,
        'concave points_mean': 0.1471,
        'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871,
        'radius_worst': 25.38,
        'texture_worst': 17.33,
        'perimeter_worst': 184.6,
        'area_worst': 2019
    }
    
    if 'breast' in predictor.models:
        breast_result = predictor.predict(
            data=breast_data,
            cancer_type='breast',
            include_staging=True,
            include_recommendations=True
        )
        
        if breast_result['success']:
            print(f"\n✓ Prediction: {breast_result['prediction']}")
            print(f"  Confidence: {breast_result['confidence']*100:.2f}%")
            print(f"  Malignant: {breast_result['is_malignant']}")
            
            if 'staging' in breast_result:
                print(f"\n  Stage: {breast_result['staging']['stage']}")
                print(f"  Risk Level: {breast_result['staging']['risk_level']}")
            
            # Generate text report
            report = predictor.generate_report(
                breast_result,
                output_format='text',
                output_file='sample_breast_cancer_report.txt'
            )
            print("\n✓ Full report saved to 'sample_breast_cancer_report.txt'")
    
    print("\n" + "="*80)
    print("Multi-Cancer Prediction System Ready!")
    print("="*80)
