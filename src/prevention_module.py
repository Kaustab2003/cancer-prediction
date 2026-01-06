"""
Comprehensive Cancer Prevention and Precaution Module
Provides personalized recommendations based on cancer type, stage, and risk factors
"""

from typing import Dict, List, Any
import json
import datetime


class PreventionAdvisor:
    """
    Comprehensive prevention and precaution advisor for multiple cancer types
    Provides evidence-based, personalized recommendations based on patient data
    """
    
    def __init__(self):
        self.recommendations_db = self._load_recommendations_database()
        self.risk_factors = self._initialize_risk_factors()
    
    def _initialize_risk_factors(self) -> Dict:
        """Initialize risk factor database for personalization"""
        return {
            'breast': ['age', 'family_history', 'brca_mutation', 'menstrual_history', 'obesity', 'alcohol', 'hrt_use'],
            'lung': ['smoking', 'smoking_years', 'radon_exposure', 'asbestos_exposure', 'family_history', 'copd'],
            'skin': ['sun_exposure', 'tanning_bed_use', 'skin_type', 'family_history', 'moles_count', 'sunburn_history'],
            'blood': ['radiation_exposure', 'chemical_exposure', 'family_history', 'genetic_disorders', 'prior_chemo']
        }
    
    def _assess_personal_risk(self, cancer_type: str, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess personalized risk based on patient data"""
        risk_score = 0
        risk_factors_present = []
        recommendations = []
        
        # Age-based risk
        age = patient_data.get('age', 0)
        if age > 0:
            if cancer_type == 'breast' and age > 50:
                risk_score += 2
                risk_factors_present.append('Advanced age (>50)')
            elif cancer_type == 'lung' and age > 55:
                risk_score += 2
                risk_factors_present.append('Advanced age (>55)')
            elif cancer_type == 'skin' and age > 40:
                risk_score += 1
                risk_factors_present.append('Age >40')
        
        # Smoking (major risk for lung cancer)
        if patient_data.get('smoking', False):
            if cancer_type == 'lung':
                risk_score += 5
                risk_factors_present.append('🚨 ACTIVE SMOKER - HIGHEST RISK FACTOR')
                recommendations.append('URGENT: Enroll in smoking cessation program immediately')
                recommendations.append('Consider nicotine replacement therapy or medications (varenicline, bupropion)')
            else:
                risk_score += 2
                risk_factors_present.append('Tobacco use')
        
        smoking_years = patient_data.get('smoking_years', 0)
        if smoking_years > 20:
            risk_score += 3
            risk_factors_present.append(f'{smoking_years} pack-years of smoking')
            recommendations.append('Qualify for low-dose CT lung cancer screening')
        
        # Family history
        if patient_data.get('family_history', False):
            risk_score += 3
            risk_factors_present.append('Family history of cancer')
            if cancer_type == 'breast':
                recommendations.append('Genetic counseling and BRCA1/BRCA2 testing recommended')
                recommendations.append('Consider starting mammogram screening 10 years before youngest family case')
            elif cancer_type == 'skin':
                recommendations.append('Annual full-body skin examination by dermatologist')
        
        # BRCA mutation
        if patient_data.get('brca_mutation', False) and cancer_type == 'breast':
            risk_score += 5
            risk_factors_present.append('🚨 BRCA1/BRCA2 mutation - 40-80% lifetime risk')
            recommendations.append('Consider prophylactic mastectomy or intensive screening')
            recommendations.append('Annual breast MRI in addition to mammogram')
            recommendations.append('Discuss chemoprevention (tamoxifen/raloxifene)')
        
        # Obesity/BMI
        bmi = patient_data.get('bmi', 0)
        if bmi > 30:
            risk_score += 2
            risk_factors_present.append(f'Obesity (BMI {bmi:.1f})')
            recommendations.append(f'Weight loss to BMI <25 can reduce cancer risk by 20-30%')
            recommendations.append('Target: Lose 5-10% body weight as initial goal')
        elif bmi > 25:
            risk_score += 1
            risk_factors_present.append(f'Overweight (BMI {bmi:.1f})')
        
        # Alcohol consumption
        alcohol = patient_data.get('alcohol_drinks_per_week', 0)
        if alcohol > 7:  # More than moderate consumption
            risk_score += 2
            risk_factors_present.append(f'Excessive alcohol ({alcohol} drinks/week)')
            recommendations.append('Reduce alcohol to ≤7 drinks/week for women, ≤14 for men')
        
        # Sun exposure (skin cancer)
        if cancer_type == 'skin':
            sun_exposure = patient_data.get('sun_exposure', 'moderate')
            if sun_exposure == 'high':
                risk_score += 3
                risk_factors_present.append('High sun exposure')
                recommendations.append('CRITICAL: Daily SPF 50+ sunscreen, avoid 10AM-4PM sun')
            
            tanning_bed = patient_data.get('tanning_bed_use', False)
            if tanning_bed:
                risk_score += 4
                risk_factors_present.append('🚨 Tanning bed use - 75% increased melanoma risk')
                recommendations.append('STOP all tanning bed use immediately')
            
            sunburns = patient_data.get('severe_sunburns', 0)
            if sunburns > 5:
                risk_score += 2
                risk_factors_present.append(f'{sunburns} severe sunburns in lifetime')
        
        # Radiation/chemical exposure
        if patient_data.get('radiation_exposure', False):
            risk_score += 3
            risk_factors_present.append('Prior radiation exposure')
        
        if patient_data.get('chemical_exposure', False):
            risk_score += 2
            risk_factors_present.append('Occupational chemical exposure')
            recommendations.append('Use proper protective equipment at work')
        
        # Calculate risk level
        if risk_score >= 10:
            calculated_risk = 'Very High'
        elif risk_score >= 7:
            calculated_risk = 'High'
        elif risk_score >= 4:
            calculated_risk = 'Moderate'
        else:
            calculated_risk = 'Low'
        
        return {
            'risk_score': risk_score,
            'risk_level': calculated_risk,
            'risk_factors_present': risk_factors_present,
            'personalized_recommendations': recommendations
        }
    
    def get_comprehensive_recommendations(
        self, 
        cancer_type: str, 
        stage: str,
        risk_level: str = None,
        patient_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive, personalized recommendations based on patient data
        
        Args:
            cancer_type: Type of cancer (breast, lung, skin, blood)
            stage: Cancer stage
            risk_level: Risk level (optional - will be calculated from patient data)
            patient_data: Patient information for personalization (age, smoking, bmi, family_history, etc.)
            
        Returns:
            Dictionary with personalized recommendations
        """
        patient_data = patient_data or {}
        
        # Perform personalized risk assessment
        risk_assessment = self._assess_personal_risk(cancer_type, patient_data)
        
        # Use calculated risk level if not provided
        if risk_level is None:
            risk_level = risk_assessment['risk_level']
        
        recommendations = {
            'cancer_type': cancer_type,
            'stage': stage,
            'risk_level': risk_level,
            'risk_assessment': risk_assessment,
            'patient_profile': self._build_patient_profile(patient_data),
            'immediate_actions': self._get_immediate_actions(cancer_type, stage, risk_level, patient_data),
            'lifestyle_modifications': self._get_lifestyle_recommendations(cancer_type, patient_data),
            'dietary_recommendations': self._get_dietary_advice(cancer_type, stage, patient_data),
            'screening_schedule': self._get_screening_schedule(cancer_type, stage, risk_level, patient_data),
            'treatment_options': self._get_treatment_pathways(cancer_type, stage),
            'prevention_strategies': self._get_prevention_strategies(cancer_type, patient_data),
            'support_resources': self._get_support_resources(cancer_type),
            'personalized_notes': self._generate_personalized_notes(cancer_type, stage, patient_data, risk_assessment),
            'warning_signs': self._get_warning_signs(cancer_type),
            'follow_up_care': self._get_follow_up_care(cancer_type, stage)
        }
        
        return recommendations
    
    def _build_patient_profile(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build a patient profile summary"""
        profile = {}
        
        if 'age' in patient_data:
            profile['age'] = patient_data['age']
        if 'gender' in patient_data:
            profile['gender'] = patient_data['gender']
        if 'bmi' in patient_data:
            bmi = patient_data['bmi']
            if bmi < 18.5:
                profile['weight_status'] = 'Underweight'
            elif bmi < 25:
                profile['weight_status'] = 'Normal'
            elif bmi < 30:
                profile['weight_status'] = 'Overweight'
            else:
                profile['weight_status'] = 'Obese'
        
        if patient_data.get('smoking', False):
            profile['smoking_status'] = 'Current smoker'
        elif patient_data.get('former_smoker', False):
            profile['smoking_status'] = 'Former smoker'
        else:
            profile['smoking_status'] = 'Never smoked'
        
        if patient_data.get('family_history', False):
            profile['family_history'] = 'Positive'
        
        return profile
    
    def _generate_personalized_notes(self, cancer_type: str, stage: str, 
                                    patient_data: Dict[str, Any], 
                                    risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate personalized notes based on patient data"""
        notes = []
        
        # Add risk-specific notes
        if risk_assessment['risk_score'] >= 10:
            notes.append('⚠️ Your risk profile indicates VERY HIGH risk - aggressive prevention measures recommended')
        elif risk_assessment['risk_score'] >= 7:
            notes.append('⚠️ Your risk profile indicates HIGH risk - intensive screening and prevention recommended')
        
        # Add modifiable risk factor notes
        if patient_data.get('smoking', False):
            notes.append('🚭 Smoking cessation is your #1 priority - can reduce risk by 30-50% within 5-10 years')
        
        bmi = patient_data.get('bmi', 0)
        if bmi > 30:
            notes.append(f'⚖️ Weight loss can significantly reduce your risk - target BMI <25 (current: {bmi:.1f})')
        
        if patient_data.get('alcohol_drinks_per_week', 0) > 7:
            notes.append('🍷 Reducing alcohol consumption can lower your cancer risk')
        
        # Add age-appropriate notes
        age = patient_data.get('age', 0)
        if cancer_type == 'breast' and age < 40 and patient_data.get('family_history', False):
            notes.append('👨‍👩‍👧 Given your family history and age, early screening is crucial')
        
        if cancer_type == 'skin' and patient_data.get('tanning_bed_use', False):
            notes.append('🛑 Tanning bed use is a major melanoma risk - discontinue immediately')
        
        return notes
    
    def _get_immediate_actions(self, cancer_type: str, stage: str, risk_level: str, 
                              patient_data: Dict[str, Any] = None) -> List[str]:
        """Get immediate actions patient should take"""
        actions = []
        
        # Universal immediate actions
        actions.append("Schedule an appointment with an oncologist within 1-2 weeks")
        actions.append("Gather all medical records and test results for review")
        
        # Stage-specific actions
        if 'IV' in stage or 'Advanced' in stage:
            actions.append("🚨 URGENT: Seek immediate medical consultation (within 24-48 hours)")
            actions.append("Prepare for potential hospitalization or aggressive treatment")
            actions.append("Arrange for caregiver support and family assistance")
        elif 'III' in stage:
            actions.append("Schedule comprehensive diagnostic imaging (CT, MRI, or PET scan)")
            actions.append("Consult with surgical oncology and radiation oncology teams")
        elif 'II' in stage:
            actions.append("Complete staging workup with recommended diagnostic tests")
            actions.append("Discuss treatment options with multidisciplinary cancer team")
        else:
            actions.append("Undergo additional diagnostic testing to confirm stage")
            actions.append("Consult with oncologist to discuss observation vs. treatment")
        
        # Cancer-specific actions
        if cancer_type.lower() == 'breast':
            actions.append("Consider genetic testing (BRCA1/BRCA2) if family history present")
            actions.append("Schedule consultation with breast surgeon")
        elif cancer_type.lower() == 'lung':
            actions.append("If smoker: Enroll in smoking cessation program immediately")
            actions.append("Undergo pulmonary function tests")
        elif cancer_type.lower() == 'skin':
            actions.append("Document all skin lesions with photographs")
            actions.append("Avoid sun exposure and use SPF 50+ sunscreen daily")
        elif cancer_type.lower() == 'blood':
            actions.append("Schedule bone marrow biopsy if not already done")
            actions.append("Consult with hematologist-oncologist")
            actions.append("Prepare for potential chemotherapy initiation")
        
        return actions
    
    def _get_lifestyle_recommendations(self, cancer_type: str, patient_data: Dict) -> Dict[str, List[str]]:
        """Get lifestyle modification recommendations"""
        recommendations = {
            'physical_activity': [],
            'stress_management': [],
            'sleep_hygiene': [],
            'substance_avoidance': [],
            'environmental_factors': []
        }
        
        # Physical activity
        recommendations['physical_activity'] = [
            "Aim for 150 minutes of moderate aerobic exercise per week",
            "Include strength training 2-3 times per week",
            "Start with gentle walking and gradually increase intensity",
            "Consider yoga or tai chi for flexibility and stress relief",
            "Consult with physical therapist for personalized exercise plan"
        ]
        
        # Stress management
        recommendations['stress_management'] = [
            "Practice mindfulness meditation for 10-15 minutes daily",
            "Join a cancer support group (in-person or online)",
            "Consider professional counseling or therapy",
            "Engage in relaxing hobbies (reading, music, art)",
            "Maintain social connections with friends and family"
        ]
        
        # Sleep hygiene
        recommendations['sleep_hygiene'] = [
            "Maintain consistent sleep schedule (7-9 hours nightly)",
            "Create dark, cool, quiet sleeping environment",
            "Avoid screens 1 hour before bedtime",
            "Limit caffeine after 2 PM",
            "Address sleep disturbances with healthcare provider"
        ]
        
        # Substance avoidance
        recommendations['substance_avoidance'] = [
            "🚭 CRITICAL: Stop all tobacco use immediately",
            "Limit alcohol consumption (no more than 1 drink/day for women, 2 for men)",
            "Avoid recreational drugs and unapproved supplements",
            "Discuss all medications with oncologist before use"
        ]
        
        # Cancer-specific recommendations
        if cancer_type.lower() == 'lung':
            recommendations['substance_avoidance'].insert(0, 
                "⚠️ PRIORITY: Smoking cessation is the single most important action")
            recommendations['environmental_factors'] = [
                "Test home for radon gas and remediate if elevated",
                "Avoid secondhand smoke exposure",
                "Use protective equipment if exposed to occupational carcinogens",
                "Improve indoor air quality with HEPA filters"
            ]
        
        elif cancer_type.lower() == 'skin':
            recommendations['environmental_factors'] = [
                "Avoid sun exposure between 10 AM - 4 PM",
                "Wear broad-spectrum SPF 50+ sunscreen daily (reapply every 2 hours)",
                "Use protective clothing (wide-brimmed hat, long sleeves, UV-blocking sunglasses)",
                "Never use tanning beds or sun lamps",
                "Seek shade and use UV-protective umbrellas outdoors"
            ]
        
        elif cancer_type.lower() == 'breast':
            recommendations['physical_activity'].insert(0,
                "Regular exercise reduces recurrence risk by 30-50%")
            recommendations['substance_avoidance'].append(
                "Minimize hormone exposure (discuss HRT with doctor)")
        
        return recommendations
    
    def _get_dietary_advice(self, cancer_type: str, stage: str, patient_data: Dict[str, Any] = None) -> Dict[str, List[str]]:
        """Get personalized dietary recommendations based on patient data"""
        patient_data = patient_data or {}
        dietary_advice = {
            'foods_to_eat': [],
            'foods_to_avoid': [],
            'supplements': [],
            'hydration': [],
            'meal_planning': [],
            'personalized_tips': []
        }
        
        # Foods to eat (anti-cancer diet)
        dietary_advice['foods_to_eat'] = [
            "🥦 Cruciferous vegetables (broccoli, cauliflower, Brussels sprouts, kale)",
            "🫐 Berries high in antioxidants (blueberries, strawberries, raspberries)",
            "🐟 Fatty fish rich in omega-3 (salmon, mackerel, sardines) - 2-3 times/week",
            "🥜 Nuts and seeds (walnuts, almonds, flaxseeds, chia seeds)",
            "🍅 Tomatoes (high in lycopene) - cooked for better absorption",
            "🧄 Garlic and onions (contain sulfur compounds)",
            "🫘 Legumes and beans (high in fiber and protein)",
            "🍊 Citrus fruits (vitamin C and flavonoids)",
            "🫒 Olive oil (extra virgin, cold-pressed)",
            "🍠 Sweet potatoes and carrots (beta-carotene)",
            "🌿 Turmeric and ginger (anti-inflammatory spices)",
            "🍵 Green tea (3-4 cups daily)"
        ]
        
        # Foods to avoid
        dietary_advice['foods_to_avoid'] = [
            "❌ Processed meats (bacon, sausage, hot dogs, deli meats)",
            "❌ Red meat (limit to <18 oz per week)",
            "❌ Refined sugars and sweetened beverages",
            "❌ Highly processed foods and fast food",
            "❌ Trans fats and hydrogenated oils",
            "❌ Excessive salt (>2300mg sodium daily)",
            "❌ Alcohol (especially for breast cancer)",
            "❌ Charred or heavily grilled meats (contain carcinogens)",
            "❌ Artificial sweeteners and food additives"
        ]
        
        # Supplements (consult oncologist before starting)
        dietary_advice['supplements'] = [
            "⚠️ Consult oncologist before taking any supplements during treatment",
            "Vitamin D3 (2000-4000 IU daily if deficient)",
            "Omega-3 fish oil (if not eating fatty fish regularly)",
            "Multivitamin (to address nutritional gaps)",
            "Probiotics (especially during/after chemotherapy)",
            "Consider: Vitamin C, selenium, zinc (under medical supervision)",
            "⚠️ AVOID high-dose antioxidants during radiation/chemotherapy"
        ]
        
        # Hydration
        dietary_advice['hydration'] = [
            "Drink 8-10 glasses (64-80 oz) of water daily",
            "Increase water intake during chemotherapy",
            "Include herbal teas (green tea, chamomile, ginger)",
            "Limit caffeinated beverages to 1-2 cups daily",
            "Avoid sugary drinks and excessive fruit juice"
        ]
        
        # Meal planning
        dietary_advice['meal_planning'] = [
            "Follow Mediterranean or plant-based diet pattern",
            "Fill half your plate with vegetables and fruits",
            "Choose whole grains over refined grains",
            "Eat smaller, more frequent meals if experiencing nausea",
            "Meal prep on good days to have healthy options available",
            "Work with registered dietitian for personalized nutrition plan"
        ]
        
        # Cancer-specific dietary advice
        if cancer_type.lower() == 'breast':
            dietary_advice['foods_to_eat'].append("🌾 Whole grains and fiber (25-30g daily)")
            dietary_advice['foods_to_avoid'].append("❌ High-fat dairy products")
            dietary_advice['foods_to_avoid'].append("❌ Soy in very high amounts (moderate soy is safe)")
        
        elif cancer_type.lower() == 'lung':
            dietary_advice['foods_to_eat'].append("🥕 Foods high in beta-carotene from whole foods (not supplements)")
            dietary_advice['supplements'].append("⚠️ AVOID beta-carotene supplements (may increase lung cancer risk in smokers)")
        
        elif cancer_type.lower() == 'blood':
            dietary_advice['foods_to_eat'].append("🥩 Iron-rich foods (lean meats, spinach, lentils)")
            dietary_advice['foods_to_eat'].append("🍊 Vitamin C foods to enhance iron absorption")
            dietary_advice['hydration'].append("Increase fluids to support blood cell production")
        
        # Personalized tips based on patient data
        bmi = patient_data.get('bmi', 0)
        if bmi > 30:
            dietary_advice['personalized_tips'].append(
                f'⚖️ Weight loss focus: Your BMI is {bmi:.1f}. Reduce portion sizes by 25% and increase vegetables to 50% of plate'
            )
            dietary_advice['meal_planning'].insert(0, 'Calorie deficit of 500 kcal/day for gradual weight loss (1 lb/week)')
        elif bmi > 25:
            dietary_advice['personalized_tips'].append(
                f'⚖️ Weight management: Your BMI is {bmi:.1f}. Maintain portion control and increase physical activity'
            )
        
        if patient_data.get('diabetes', False):
            dietary_advice['personalized_tips'].append('🍭 Diabetes: Choose low glycemic index foods, limit refined carbohydrates')
            dietary_advice['foods_to_avoid'].append('❌ High-sugar foods and refined carbohydrates (manage blood glucose)')
        
        if patient_data.get('alcohol_drinks_per_week', 0) > 7:
            dietary_advice['personalized_tips'].append(
                f'🍷 Reduce alcohol from {patient_data["alcohol_drinks_per_week"]} drinks/week to ≤7 drinks/week'
            )
        
        return dietary_advice
    
    def _get_screening_schedule(self, cancer_type: str, stage: str, risk_level: str, 
                               patient_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get personalized screening schedule based on patient risk factors"""
        patient_data = patient_data or {}
        schedule = {
            'initial_tests': [],
            'follow_up_frequency': '',
            'imaging_schedule': [],
            'lab_tests': [],
            'self_monitoring': [],
            'personalized_schedule': []
        }
        
        if cancer_type.lower() == 'breast':
            schedule['initial_tests'] = [
                "Mammogram (bilateral)",
                "Breast ultrasound",
                "Breast MRI (if high-risk)",
                "Biopsy confirmation of diagnosis",
                "Genetic testing (BRCA1/BRCA2, PALB2)"
            ]
            schedule['follow_up_frequency'] = "Every 3-6 months for first 3 years, then annually"
            schedule['imaging_schedule'] = [
                "Mammogram: Every 6-12 months",
                "Breast MRI: Annually (for high-risk patients)",
                "CT/PET scan: As clinically indicated"
            ]
            schedule['lab_tests'] = [
                "Tumor markers (CA 15-3, CA 27-29): Every 3-6 months",
                "Complete blood count (CBC): Quarterly",
                "Liver function tests: Every 6 months"
            ]
            schedule['self_monitoring'] = [
                "Monthly breast self-examination",
                "Track any new lumps or changes",
                "Monitor for bone pain, persistent cough, or headaches"
            ]
        
        elif cancer_type.lower() == 'lung':
            schedule['initial_tests'] = [
                "Low-dose CT scan of chest",
                "PET-CT scan for staging",
                "Bronchoscopy with biopsy",
                "Pulmonary function tests (PFTs)",
                "Molecular testing (EGFR, ALK, ROS1, PD-L1)"
            ]
            schedule['follow_up_frequency'] = "Every 3-4 months for first 2 years, then every 6 months"
            schedule['imaging_schedule'] = [
                "Chest CT: Every 3-6 months for 2 years, then annually",
                "PET-CT: As clinically indicated",
                "Brain MRI: Annually or if symptoms develop"
            ]
            schedule['lab_tests'] = [
                "Tumor markers (CEA, CYFRA 21-1): Every 3 months",
                "Complete blood count: Quarterly",
                "Comprehensive metabolic panel: Every 6 months"
            ]
            schedule['self_monitoring'] = [
                "Monitor respiratory symptoms (cough, shortness of breath)",
                "Track unexplained weight loss",
                "Note any chest pain or hemoptysis (coughing blood)"
            ]
        
        elif cancer_type.lower() == 'skin':
            schedule['initial_tests'] = [
                "Full-body skin examination by dermatologist",
                "Dermoscopy of suspicious lesions",
                "Skin biopsy for histopathological diagnosis",
                "Sentinel lymph node biopsy (for melanoma ≥T1b)",
                "PET-CT or MRI (for advanced melanoma)"
            ]
            schedule['follow_up_frequency'] = "Every 3-6 months for melanoma, annually for non-melanoma"
            schedule['imaging_schedule'] = [
                "Full-body skin exam: Every 3-6 months for 5 years",
                "Lymph node ultrasound: Every 6 months (if indicated)",
                "CT or PET-CT: Annually for Stage II-III melanoma"
            ]
            schedule['lab_tests'] = [
                "LDH (lactate dehydrogenase): Every 3-6 months for melanoma",
                "Complete blood count: Every 6 months",
                "Liver function tests: Annually"
            ]
            schedule['self_monitoring'] = [
                "🔍 Monthly skin self-examination (ABCDE rule)",
                "A: Asymmetry, B: Border irregularity, C: Color variation",
                "D: Diameter >6mm, E: Evolution/Changing",
                "Photograph all moles for comparison",
                "Check family members' skin"
            ]
        
        elif cancer_type.lower() == 'blood':
            schedule['initial_tests'] = [
                "Bone marrow aspiration and biopsy",
                "Complete blood count with differential",
                "Flow cytometry for immunophenotyping",
                "Cytogenetic analysis and molecular testing",
                "Lumbar puncture (CNS evaluation)"
            ]
            schedule['follow_up_frequency'] = "Weekly during induction, then monthly during maintenance"
            schedule['imaging_schedule'] = [
                "Bone marrow biopsy: At diagnosis, post-induction, and as needed",
                "CT scan: Only if specific symptoms warrant",
                "Echocardiogram: Before anthracycline therapy"
            ]
            schedule['lab_tests'] = [
                "Complete blood count: Daily to weekly (depending on treatment phase)",
                "Bone marrow evaluation: Post-induction, every 3 months during maintenance",
                "Minimal residual disease (MRD) testing: Post-induction and periodically",
                "Liver and kidney function: Weekly during intensive treatment"
            ]
            schedule['self_monitoring'] = [
                "Monitor for fever >100.4°F (sign of infection)",
                "Check for easy bruising or bleeding",
                "Report fatigue, weakness, or bone pain",
                "Watch for enlarged lymph nodes"
            ]
        
        return schedule
    
    def _get_treatment_pathways(self, cancer_type: str, stage: str) -> Dict[str, List[str]]:
        """Get treatment options based on cancer type and stage"""
        treatments = {
            'primary_treatment': [],
            'adjuvant_therapy': [],
            'targeted_therapy': [],
            'immunotherapy': [],
            'supportive_care': []
        }
        
        if cancer_type.lower() == 'breast':
            if 'I' in stage or stage == 'Stage 0':
                treatments['primary_treatment'] = [
                    "Lumpectomy (breast-conserving surgery) with sentinel lymph node biopsy",
                    "Mastectomy (if tumor is large or multifocal)",
                    "Radiation therapy (5-6 weeks post-lumpectomy)"
                ]
                treatments['adjuvant_therapy'] = [
                    "Hormone therapy (tamoxifen or aromatase inhibitors for 5-10 years)",
                    "Chemotherapy (if high-risk features present)",
                    "Consider oncotype DX score to guide chemotherapy decision"
                ]
            
            elif 'II' in stage or 'III' in stage:
                treatments['primary_treatment'] = [
                    "Neoadjuvant chemotherapy (to shrink tumor before surgery)",
                    "Mastectomy with axillary lymph node dissection",
                    "Breast reconstruction (immediate or delayed)",
                    "Radiation therapy post-surgery"
                ]
                treatments['adjuvant_therapy'] = [
                    "Chemotherapy (AC-T, TAC, or dose-dense regimens)",
                    "Hormone therapy for hormone receptor-positive tumors",
                    "HER2-targeted therapy (trastuzumab/pertuzumab for HER2+ tumors)"
                ]
                treatments['targeted_therapy'] = [
                    "Trastuzumab (Herceptin) for HER2-positive (1 year)",
                    "Pertuzumab (Perjeta) for advanced HER2-positive",
                    "CDK4/6 inhibitors (palbociclib, ribociclib) for HR+/HER2-",
                    "PARP inhibitors (olaparib, talazoparib) for BRCA mutations"
                ]
            
            elif 'IV' in stage:
                treatments['primary_treatment'] = [
                    "Systemic chemotherapy (primary treatment modality)",
                    "Palliative surgery/radiation for symptom control",
                    "Consider clinical trials for novel therapies"
                ]
                treatments['targeted_therapy'] = [
                    "HER2-targeted therapy combinations",
                    "CDK4/6 inhibitors + hormone therapy (first-line for HR+)",
                    "mTOR inhibitors (everolimus)",
                    "PI3K inhibitors (alpelisib for PIK3CA mutations)"
                ]
                treatments['immunotherapy'] = [
                    "Pembrolizumab (for PD-L1 positive triple-negative)",
                    "Atezolizumab + nab-paclitaxel (for metastatic triple-negative)"
                ]
        
        elif cancer_type.lower() == 'lung':
            if 'I' in stage:
                treatments['primary_treatment'] = [
                    "Surgical resection (lobectomy or pneumonectomy)",
                    "Stereotactic body radiation therapy (SBRT) if not surgical candidate",
                    "Video-assisted thoracoscopic surgery (VATS)"
                ]
                treatments['adjuvant_therapy'] = [
                    "Consider adjuvant chemotherapy for tumors >4cm",
                    "Immunotherapy (for Stage IB-IIIA)"
                ]
            
            elif 'II' in stage or 'III' in stage:
                treatments['primary_treatment'] = [
                    "Concurrent chemoradiation (definitive or neoadjuvant)",
                    "Surgical resection + mediastinal lymph node dissection",
                    "Consolidation immunotherapy (durvalumab) post-chemoradiation"
                ]
                treatments['adjuvant_therapy'] = [
                    "Platinum-based chemotherapy (cisplatin/carboplatin + etoposide/pemetrexed)",
                    "Immunotherapy (pembrolizumab, atezolizumab) for 1 year",
                    "Radiation therapy (60-70 Gy)"
                ]
            
            elif 'IV' in stage:
                treatments['primary_treatment'] = [
                    "Systemic therapy (chemotherapy, targeted therapy, or immunotherapy)",
                    "Palliative radiation for symptomatic metastases",
                    "Clinical trials for novel combinations"
                ]
                treatments['targeted_therapy'] = [
                    "EGFR inhibitors (osimertinib, erlotinib) for EGFR mutations",
                    "ALK inhibitors (alectinib, brigatinib) for ALK rearrangements",
                    "ROS1 inhibitors (crizotinib, entrectinib) for ROS1 fusions",
                    "KRAS G12C inhibitors (sotorasib, adagrasib) for KRAS mutations"
                ]
                treatments['immunotherapy'] = [
                    "PD-1/PD-L1 inhibitors (pembrolizumab, nivolumab, atezolizumab)",
                    "Combination immunotherapy (nivolumab + ipilimumab)",
                    "Chemo-immunotherapy combinations"
                ]
        
        elif cancer_type.lower() == 'skin':
            if 'melanoma' in stage.lower() or 'I' in stage or 'II' in stage:
                treatments['primary_treatment'] = [
                    "Wide local excision with appropriate margins",
                    "Sentinel lymph node biopsy (for melanoma ≥T1b)",
                    "Mohs micrographic surgery (for high-risk or facial lesions)"
                ]
                treatments['adjuvant_therapy'] = [
                    "Adjuvant immunotherapy (pembrolizumab or nivolumab) for Stage IIB-III",
                    "Targeted therapy (dabrafenib + trametinib for BRAF V600 mutations)"
                ]
            
            elif 'III' in stage:
                treatments['primary_treatment'] = [
                    "Wide excision + complete lymph node dissection",
                    "Consider neoadjuvant immunotherapy or targeted therapy"
                ]
                treatments['adjuvant_therapy'] = [
                    "Adjuvant immunotherapy (1 year): pembrolizumab or nivolumab",
                    "BRAF/MEK inhibitors for BRAF-mutant melanoma"
                ]
            
            treatments['immunotherapy'] = [
                "Anti-PD-1 (pembrolizumab, nivolumab)",
                "Anti-CTLA-4 (ipilimumab)",
                "Combination: nivolumab + ipilimumab (for advanced disease)",
                "Intralesional therapy (T-VEC) for injectable lesions"
            ]
            
            treatments['targeted_therapy'] = [
                "BRAF inhibitors (dabrafenib, vemurafenib, encorafenib)",
                "MEK inhibitors (trametinib, cobimetinib, binimetinib)",
                "BRAF + MEK combination therapy (preferred over monotherapy)"
            ]
        
        elif cancer_type.lower() == 'blood':
            treatments['primary_treatment'] = [
                "Induction chemotherapy (multi-drug regimen): 4-6 weeks",
                "Consolidation chemotherapy: 4-8 months",
                "Maintenance therapy: Up to 2-3 years",
                "CNS prophylaxis (intrathecal chemotherapy)"
            ]
            treatments['adjuvant_therapy'] = [
                "Allogeneic stem cell transplant (for high-risk or relapsed ALL)",
                "CAR T-cell therapy (tisagenlecleucel for B-cell ALL)",
                "Radiation therapy (rarely, for CNS disease or emergencies)"
            ]
            treatments['targeted_therapy'] = [
                "Tyrosine kinase inhibitors (imatinib, dasatinib for Ph+ ALL)",
                "Blinatumomab (BiTE antibody for B-cell ALL)",
                "Inotuzumab ozogamicin (antibody-drug conjugate for CD22+ ALL)",
                "Venetoclax (BCL-2 inhibitor, in combinations)"
            ]
            treatments['immunotherapy'] = [
                "CAR T-cell therapy (tisagenlecleucel for relapsed/refractory B-ALL)",
                "Blinatumomab (immunotherapy engaging T-cells)"
            ]
        
        # Universal supportive care
        treatments['supportive_care'] = [
            "Pain management (multimodal approach)",
            "Anti-nausea medications (ondansetron, aprepitant)",
            "Growth factors (G-CSF, erythropoietin) for blood counts",
            "Nutritional support and counseling",
            "Physical therapy and rehabilitation",
            "Psychosocial support and counseling",
            "Palliative care consultation (for symptom management)",
            "Fertility preservation discussion (before treatment)",
            "Clinical trial enrollment consideration"
        ]
        
        return treatments
    
    def _get_prevention_strategies(self, cancer_type: str, patient_data: Dict) -> List[str]:
        """Get prevention strategies for reducing cancer risk"""
        strategies = []
        
        # Universal prevention
        strategies.extend([
            "🏃 Maintain healthy weight (BMI 18.5-24.9)",
            "🥗 Eat a balanced diet rich in fruits, vegetables, and whole grains",
            "💪 Exercise regularly (150+ minutes moderate activity per week)",
            "🚭 Avoid all tobacco products",
            "🍷 Limit alcohol consumption",
            "😴 Get adequate sleep (7-9 hours nightly)",
            "🧘 Manage stress through meditation, yoga, or counseling",
            "💉 Stay up-to-date with vaccinations (HPV, Hepatitis B)",
            "🧪 Participate in recommended cancer screening programs"
        ])
        
        # Cancer-specific prevention
        if cancer_type.lower() == 'breast':
            strategies.extend([
                "🤱 Breastfeed if possible (reduces risk by 20%)",
                "💊 Consider chemoprevention (tamoxifen/raloxifene) if high-risk",
                "🧬 Genetic counseling and testing if strong family history",
                "🏥 Annual mammograms starting at age 40 (earlier if high-risk)",
                "🔍 Monthly breast self-exams and annual clinical exams"
            ])
        
        elif cancer_type.lower() == 'lung':
            strategies.extend([
                "🚭 CRITICAL: Quit smoking (reduces risk by 30-50% within 10 years)",
                "🏠 Test home for radon and remediate if necessary",
                "🏭 Avoid occupational carcinogens (asbestos, diesel exhaust)",
                "😷 Use protective equipment in high-risk occupations",
                "🔬 Low-dose CT screening if age 50-80 with 20+ pack-year history"
            ])
        
        elif cancer_type.lower() == 'skin':
            strategies.extend([
                "☀️ Avoid peak sun hours (10 AM - 4 PM)",
                "🧴 Apply broad-spectrum SPF 50+ sunscreen daily",
                "👒 Wear protective clothing, hats, and UV-blocking sunglasses",
                "❌ NEVER use tanning beds or sun lamps",
                "🔍 Perform monthly skin self-exams (ABCDE rule)",
                "🏥 Annual full-body skin exam by dermatologist (if high-risk)"
            ])
        
        elif cancer_type.lower() == 'blood':
            strategies.extend([
                "☢️ Avoid unnecessary radiation exposure",
                "🧪 Minimize exposure to benzene and other industrial chemicals",
                "🩸 Monitor blood counts if receiving chemotherapy for other cancers",
                "🧬 Genetic counseling if family history of leukemia",
                "💉 Avoid infections during immunosuppression"
            ])
        
        return strategies
    
    def _get_support_resources(self, cancer_type: str) -> Dict[str, List[str]]:
        """Get support resources and organizations"""
        resources = {
            'organizations': [
                "American Cancer Society (1-800-227-2345, cancer.org)",
                "National Cancer Institute (1-800-4-CANCER, cancer.gov)",
                "CancerCare (1-800-813-HOPE, cancercare.org)",
                "Cancer Support Community (cancersupportcommunity.org)"
            ],
            'financial_assistance': [
                "Patient Access Network Foundation (panfoundation.org)",
                "HealthWell Foundation (healthwellfoundation.org)",
                "CancerCare Co-Payment Assistance Foundation",
                "Pharmaceutical company patient assistance programs",
                "Social Security Disability Insurance (SSDI) application"
            ],
            'online_communities': [
                "Cancer Survivors Network (csn.cancer.org)",
                "Inspire Cancer Community (inspire.com)",
                "Smart Patients (smartpatients.com)",
                "Cancer subreddit (reddit.com/r/cancer)"
            ],
            'caregiver_support': [
                "Family Caregiver Alliance (caregiver.org)",
                "Cancer Care for Caregivers (cancercare.org/caregivers)",
                "LIVESTRONG at the YMCA (exercise program)",
                "Local hospital support groups and counseling"
            ]
        }
        
        # Cancer-specific resources
        if cancer_type.lower() == 'breast':
            resources['organizations'].extend([
                "Susan G. Komen Foundation (komen.org)",
                "Breastcancer.org",
                "Living Beyond Breast Cancer (lbbc.org)",
                "Young Survival Coalition (for women <40)"
            ])
        
        elif cancer_type.lower() == 'lung':
            resources['organizations'].extend([
                "Lung Cancer Foundation of America (lcfamerica.org)",
                "LUNGevity Foundation (lungevity.org)",
                "GO2 Foundation for Lung Cancer (go2foundation.org)",
                "American Lung Association (lung.org)"
            ])
        
        elif cancer_type.lower() == 'skin':
            resources['organizations'].extend([
                "Melanoma Research Foundation (melanoma.org)",
                "Skin Cancer Foundation (skincancer.org)",
                "American Academy of Dermatology (aad.org)",
                "AIM at Melanoma Foundation (aimatmelanoma.org)"
            ])
        
        elif cancer_type.lower() == 'blood':
            resources['organizations'].extend([
                "Leukemia & Lymphoma Society (lls.org, 1-800-955-4572)",
                "National Marrow Donor Program (BeTheMatch.org)",
                "Children's Oncology Group (for pediatric ALL)",
                "Lymphoma Research Foundation (lymphoma.org)"
            ])
        
        return resources
    
    def _get_warning_signs(self, cancer_type: str) -> List[str]:
        """Get warning signs that require immediate medical attention"""
        warning_signs = {
            'breast': [
                "🚨 New lump or mass in breast or underarm",
                "🚨 Changes in breast size or shape",
                "🚨 Nipple discharge (especially bloody)",
                "🚨 Nipple inversion or retraction",
                "🚨 Skin changes (dimpling, puckering, redness, scaling)",
                "🚨 Persistent breast pain in one area"
            ],
            'lung': [
                "🚨 Persistent cough lasting >3 weeks",
                "🚨 Coughing up blood (hemoptysis)",
                "🚨 Severe shortness of breath",
                "🚨 Chest pain that worsens with deep breathing",
                "🚨 Hoarseness lasting >2 weeks",
                "🚨 Unexplained weight loss >10 lbs",
                "🚨 Recurrent pneumonia or bronchitis"
            ],
            'skin': [
                "🚨 Mole changing in size, shape, or color",
                "🚨 New pigmented or unusual-looking growth",
                "🚨 Sore that doesn't heal within 3 weeks",
                "🚨 Mole that itches, bleeds, or is painful",
                "🚨 Spread of pigmentation beyond border",
                "🚨 Any ABCDE criteria: Asymmetry, Border, Color, Diameter, Evolution"
            ],
            'blood': [
                "🚨 Fever >100.4°F (especially if neutropenic)",
                "🚨 Severe bleeding or bruising",
                "🚨 Extreme fatigue or weakness",
                "🚨 Severe headache or vision changes",
                "🚨 Difficulty breathing or chest pain",
                "🚨 Confusion or altered mental status",
                "🚨 Severe bone or joint pain"
            ]
        }
        
        general_warning = [
            "⚠️ If experiencing any of these symptoms, contact your oncologist immediately",
            "⚠️ Go to emergency room for severe symptoms (difficulty breathing, chest pain, severe bleeding)",
            "⚠️ Have emergency contact information readily available"
        ]
        
        cancer_signs = warning_signs.get(cancer_type.lower(), [])
        return cancer_signs + general_warning
    
    def _get_follow_up_care(self, cancer_type: str, stage: str) -> Dict[str, Any]:
        """Get follow-up care recommendations"""
        follow_up = {
            'surveillance_period': '',
            'key_appointments': [],
            'survivorship_care': [],
            'late_effects_monitoring': []
        }
        
        # Surveillance period
        if 'IV' in stage or 'Advanced' in stage:
            follow_up['surveillance_period'] = "Indefinite - ongoing treatment and monitoring"
        elif 'III' in stage:
            follow_up['surveillance_period'] = "Intensive: 5 years, then annually for life"
        else:
            follow_up['surveillance_period'] = "Intensive: 3-5 years, then annually"
        
        # Key appointments
        follow_up['key_appointments'] = [
            "Oncology visits: Every 3-6 months initially, gradually extending",
            "Primary care coordination for overall health",
            "Survivorship care plan review annually",
            "Dental exam before chemotherapy/radiation",
            "Ophthalmology exam (if certain chemotherapies used)",
            "Cardiology follow-up (if anthracyclines or trastuzumab used)"
        ]
        
        # Survivorship care
        follow_up['survivorship_care'] = [
            "Request written survivorship care plan from oncologist",
            "Maintain healthy lifestyle (diet, exercise, no smoking)",
            "Attend survivorship programs at cancer center",
            "Address psychosocial needs (anxiety, depression, fear of recurrence)",
            "Manage long-term side effects of treatment",
            "Stay current with other health screenings (colonoscopy, etc.)"
        ]
        
        # Late effects monitoring
        if cancer_type.lower() == 'breast':
            follow_up['late_effects_monitoring'] = [
                "Cardiac toxicity (from anthracyclines or trastuzumab) - annual ECG/echo",
                "Lymphedema management and prevention",
                "Bone health monitoring (DEXA scans for aromatase inhibitor users)",
                "Menopausal symptoms management",
                "Cognitive changes ('chemo brain')",
                "Second cancer screening (contralateral breast, ovarian if BRCA+)"
            ]
        
        elif cancer_type.lower() == 'lung':
            follow_up['late_effects_monitoring'] = [
                "Pulmonary function decline - annual PFTs",
                "Cardiac toxicity (from radiation or certain chemotherapies)",
                "Esophageal stricture (from radiation)",
                "Peripheral neuropathy (from platinum-based chemotherapy)",
                "Hearing loss (from cisplatin)",
                "Second primary lung cancer screening"
            ]
        
        elif cancer_type.lower() == 'skin':
            follow_up['late_effects_monitoring'] = [
                "Development of additional skin cancers - lifelong screening",
                "Lymphedema (if lymph node dissection performed)",
                "Immune-related adverse events (if immunotherapy used)",
                "Psychological impact and body image concerns"
            ]
        
        elif cancer_type.lower() == 'blood':
            follow_up['late_effects_monitoring'] = [
                "Infection risk monitoring - annual vaccines",
                "Cardiac toxicity (from anthracyclines) - echocardiograms",
                "Endocrine dysfunction - thyroid function tests",
                "Cognitive effects - neuropsychological testing if needed",
                "Growth and development (pediatric patients)",
                "Secondary malignancies - ongoing vigilance",
                "Graft-versus-host disease (if stem cell transplant)"
            ]
        
        return follow_up
    
    def _load_recommendations_database(self) -> Dict:
        """Load comprehensive recommendations database (placeholder for future expansion)"""
        # This could be expanded to load from external JSON file or database
        return {}


# Example usage
if __name__ == "__main__":
    advisor = PreventionAdvisor()
    
    # Example with detailed patient data for personalized recommendations
    patient_info = {
        'age': 58,
        'gender': 'female',
        'bmi': 32.5,  # Obese
        'smoking': False,
        'former_smoker': True,
        'smoking_years': 15,
        'family_history': True,  # Mother had breast cancer
        'brca_mutation': False,
        'alcohol_drinks_per_week': 10,  # Moderate-heavy
        'diabetes': False,
        'exercise_minutes_per_week': 60  # Below recommended
    }
    
    # Get personalized recommendations
    recommendations = advisor.get_comprehensive_recommendations(
        cancer_type='breast',
        stage='Stage II',
        patient_data=patient_info
    )
    
    print("="*80)
    print("PERSONALIZED CANCER CARE RECOMMENDATIONS")
    print("="*80)
    
    # Display patient profile
    print("\n👤 PATIENT PROFILE:")
    profile = recommendations['patient_profile']
    for key, value in profile.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Display risk assessment
    print("\n⚠️ RISK ASSESSMENT:")
    risk = recommendations['risk_assessment']
    print(f"  Risk Level: {risk['risk_level']} (Score: {risk['risk_score']}/15+)")
    print(f"  Risk Factors Identified: {len(risk['risk_factors_present'])}")
    for factor in risk['risk_factors_present']:
        print(f"    • {factor}")
    
    # Display personalized notes
    if recommendations.get('personalized_notes'):
        print("\n💡 PERSONALIZED INSIGHTS:")
        for note in recommendations['personalized_notes']:
            print(f"  {note}")
    
    # Display personalized recommendations from risk assessment
    if risk.get('personalized_recommendations'):
        print("\n🎯 PRIORITY ACTIONS FOR YOU:")
        for rec in risk['personalized_recommendations']:
            print(f"  • {rec}")
    
    print("\n📋 IMMEDIATE ACTIONS:")
    for action in recommendations['immediate_actions'][:5]:  # First 5
        print(f"  • {action}")
    
    print("\n💪 KEY LIFESTYLE MODIFICATIONS:")
    lifestyle = recommendations['lifestyle_modifications']
    if 'physical_activity' in lifestyle:
        print(f"  Exercise: {lifestyle['physical_activity'][0]}")
    if 'substance_avoidance' in lifestyle:
        print(f"  Alcohol: {lifestyle['substance_avoidance'][1]}")
    
    print("\n🥗 DIETARY HIGHLIGHTS:")
    diet = recommendations['dietary_recommendations']
    if diet.get('personalized_tips'):
        print("  Personalized for You:")
        for tip in diet['personalized_tips']:
            print(f"    {tip}")
    print(f"  Foods to Emphasize: {len(diet['foods_to_eat'])} recommendations")
    print(f"  Foods to Limit: {len(diet['foods_to_avoid'])} items")
    
    print("\n📅 SCREENING SCHEDULE:")
    screening = recommendations['screening_schedule']
    print(f"  Follow-up: {screening['follow_up_frequency']}")
    if screening.get('personalized_schedule'):
        print("  Personalized Schedule:")
        for item in screening['personalized_schedule']:
            print(f"    • {item}")
    
    print("\n🆘 SUPPORT RESOURCES:")
    resources = recommendations['support_resources']
    for org in resources['organizations'][:3]:
        print(f"  • {org}")
    
    print("\n" + "="*80)
    print(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    # Save personalized recommendations
    with open('personalized_prevention_plan.json', 'w') as f:
        # Convert for JSON serialization
        json.dump(recommendations, f, indent=2, default=str)
    print("\n✅ Your personalized prevention plan saved to 'personalized_prevention_plan.json'")

