"""
LLM Agent - Natural Language Explanation Generator

Uses OpenAI GPT to convert ensemble predictions and SHAP evidence
into comprehensive, actionable security alerts.

Format: Alert → Evidence → Impact → Recommendation
"""

import json
import os
from openai import OpenAI
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_api_key_from_env():
    """
    Load OpenAI API key from .env file or environment variable
    """
    # Try to load from .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'OPENAI_API_KEY':
                        return value.strip()
    
    # Try to load from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    return None

class LLMAgent:
    """
    LLM-powered agent for generating natural language security explanations
    and interactive chat
    """
    
    def __init__(self, api_key=None):
        """
        Initialize with OpenAI API key (loads from .env if not provided)
        """
        if api_key is None:
            api_key = load_api_key_from_env()
        
        if api_key is None or api_key == 'your_openai_api_key_here':
            raise ValueError(
                "OpenAI API key not found. Please:\n"
                "1. Create a .env file in the project root\n"
                "2. Add: OPENAI_API_KEY=your_actual_key\n"
                "3. Get key from: https://platform.openai.com/api-keys"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using cost-effective model
        self.chat_history = []  # For conversational chat
        self.current_prediction_context = None  # Current prediction being discussed
        
    def generate_explanation(self, ensemble_result):
        """
        Generate comprehensive natural language explanation from ensemble result
        """
        # Build prompt
        prompt = self._build_prompt(ensemble_result)
        
        # Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Low temperature for consistent, factual output
                max_tokens=1500
            )
            
            explanation = response.choices[0].message.content
            return explanation
            
        except Exception as e:
            return f"ERROR: Failed to generate LLM explanation: {str(e)}"
    
    def _get_system_prompt(self):
        """
        System prompt defining LLM behavior
        """
        return """You are an expert cybersecurity analyst AI assistant. Your job is to analyze network anomaly detection results from a dual-model ensemble system (XGBoost + LSTM) and generate clear, actionable security alerts.

CRITICAL REQUIREMENTS:
1. Be CONCISE and SPECIFIC - no fluff or generic statements
2. Focus on EVIDENCE from the ML models (SHAP values show what matters)
3. Provide ACTIONABLE recommendations
4. Use the EXACT format: Alert → Evidence → Impact → Recommendation
5. When models disagree, explain BOTH perspectives with evidence
6. Mention specific feature values and their significance
7. Keep technical but accessible for security operations team

OUTPUT FORMAT (STRICT):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚨 ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Threat Level] [Attack Type] detected with [confidence]% confidence
Timestamp: [time]
Source: [IP/Asset info]

🔍 EVIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model Agreement: [AGREED/DISAGREED]
[If disagreed, show both predictions with evidence]

Key Indicators (from SHAP analysis):
• [Feature 1]: [value] → [why this matters]
• [Feature 2]: [value] → [why this matters]
• [Feature 3]: [value] → [why this matters]

Traffic Characteristics:
• Volume: [packet rate, byte rate]
• Protocol: [details]
• Duration: [time]
• Behavior: [what's anomalous]

⚠️ IMPACT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Describe potential business/security impact]
[What systems/data are at risk]
[Severity assessment]

✅ RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMMEDIATE:
1. [Specific blocking action]
2. [Monitoring action]
3. [Investigation action]

SHORT-TERM:
• [Preventive measure 1]
• [Preventive measure 2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remember: Be specific, use the evidence, and give actionable advice."""
    
    def _build_prompt(self, result):
        """
        Build user prompt from ensemble result
        """
        ens = result['ensemble_decision']
        agents = result['agent_predictions']
        meta = result['metadata']
        evidence = result['evidence']
        
        prompt = f"""Analyze this network anomaly detection result and generate a security alert:

═══════════════════════════════════════════════════════════════════════════════
ENSEMBLE DETECTION RESULT
═══════════════════════════════════════════════════════════════════════════════

DETECTION SUMMARY:
• Timestamp: {result['timestamp']}
• Final Prediction: {ens['final_prediction']}
• Confidence: {ens['confidence']*100:.1f}%
• Agreement Status: {ens['agreement']}
• Confidence Level: {ens['confidence_level']}

SOURCE INFORMATION:
• Source IP: {meta['source_ip']}
• Affected Assets: {meta['affected_assets']}

DUAL-MODEL PREDICTIONS:
┌─────────────┬────────────────────┬──────────────┐
│ Model       │ Prediction         │ Confidence   │
├─────────────┼────────────────────┼──────────────┤
│ XGBoost     │ {agents['xgboost']['prediction']:<18} │ {agents['xgboost']['confidence']*100:>6.1f}%     │
│ LSTM        │ {agents['lstm']['prediction']:<18} │ {agents['lstm']['confidence']*100:>6.1f}%     │
└─────────────┴────────────────────┴──────────────┘

KEY NETWORK INDICATORS:
• Traffic Volume: {evidence['key_indicators']['traffic_volume']}
• Data Rate: {evidence['key_indicators']['data_rate']}
• Duration: {evidence['key_indicators']['duration']}
• Protocol: {evidence['key_indicators']['protocol']}
• TCP Flags: {evidence['key_indicators']['tcp_flags']}
• Destination Port: {evidence['key_indicators']['destination_port']}
• Traffic Intensity: {evidence['key_indicators']['traffic_intensity']}

XGBOOST MODEL EVIDENCE (Tree-based ML):
Prediction: {evidence['xgboost_evidence']['prediction']} ({evidence['xgboost_evidence']['confidence']*100:.1f}%)
Top Features by SHAP Impact:
"""
        
        for i, feat in enumerate(evidence['xgboost_evidence']['top_5_features'], 1):
            direction = "↑" if feat['shap_impact'] > 0 else "↓"
            prompt += f"{i}. {feat['feature']}: {feat['value']:.4f} {direction} SHAP={feat['shap_impact']:.4f}\n"
        
        prompt += f"\nLSTM MODEL EVIDENCE (Sequential Deep Learning):\n"
        prompt += f"Prediction: {evidence['lstm_evidence']['prediction']} ({evidence['lstm_evidence']['confidence']*100:.1f}%)\n"
        prompt += f"Top Features by SHAP Impact:\n"
        
        for i, feat in enumerate(evidence['lstm_evidence']['top_5_features'], 1):
            direction = "↑" if feat['shap_impact'] > 0 else "↓"
            prompt += f"{i}. {feat['feature']}: {feat['value']:.4f} {direction} SHAP={feat['shap_impact']:.4f}\n"
        
        if evidence['combined_top_features']:
            prompt += f"\nFEATURES WHERE BOTH MODELS AGREE:\n"
            for feat in evidence['combined_top_features'][:5]:
                agree = "✓" if feat['both_agree'] else "✗"
                prompt += f"{agree} {feat['feature']}: {feat['value']:.4f} (XGB SHAP: {feat['xgboost_shap']:.4f}, LSTM SHAP: {feat['lstm_shap']:.4f})\n"
        
        prompt += f"\n{'='*79}\n"
        prompt += f"\nGenerate a comprehensive security alert following the specified format."
        
        return prompt
    
    def process_ensemble_results(self, ensemble_results):
        """
        Process multiple ensemble results and generate explanations
        """
        print("\n" + "=" * 80)
        print("LLM AGENT - GENERATING NATURAL LANGUAGE EXPLANATIONS")
        print("=" * 80)
        
        llm_results = []
        
        for i, result in enumerate(ensemble_results, 1):
            print(f"\n📝 Generating explanation {i}/{len(ensemble_results)}...")
            print(f"   Detection: {result['ensemble_decision']['final_prediction']}")
            
            explanation = self.generate_explanation(result)
            
            llm_result = {
                'ensemble_result': result,
                'llm_explanation': explanation,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            llm_results.append(llm_result)
            
            print(f"   ✅ Explanation generated ({len(explanation)} chars)")
        
        return llm_results
    
    def save_results(self, llm_results, filename='llm_explanations.json'):
        """
        Save LLM results to JSON
        """
        with open(filename, 'w') as f:
            json.dump(llm_results, f, indent=2)
        
        print(f"\n✅ LLM explanations saved to {filename}")
    
    def save_readable_report(self, llm_results, filename='security_alert_report.txt'):
        """
        Save human-readable report
        """
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-AGENT SECURITY ALERT REPORT\n")
            f.write("Generated by: XGBoost + LSTM Ensemble with SHAP Explainability\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(llm_results, 1):
                ens = result['ensemble_result']['ensemble_decision']
                
                f.write("=" * 80 + "\n")
                f.write(f"ALERT #{i}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Detection: {ens['final_prediction']} ({ens['confidence']*100:.1f}%)\n")
                f.write(f"Agreement: {ens['agreement']}\n")
                f.write(f"Generated: {result['generated_at']}\n")
                f.write("-" * 80 + "\n\n")
                
                f.write(result['llm_explanation'])
                f.write("\n\n")
        
        print(f"✅ Readable report saved to {filename}")
    
    def print_sample_explanation(self, llm_result):
        """
        Print a sample explanation to console
        """
        print("\n" + "=" * 80)
        print("SAMPLE SECURITY ALERT")
        print("=" * 80)
        print(llm_result['llm_explanation'])
        print("=" * 80)
    
    def set_prediction_context(self, ensemble_result):
        """
        Set the current prediction context for chat conversations
        Clears previous chat history when new prediction is set
        """
        self.current_prediction_context = ensemble_result
        self.chat_history = []  # Clear previous chat
    
    def chat(self, user_message):
        """
        Interactive chat about the current prediction
        User can ask questions about the detection results
        """
        if self.current_prediction_context is None:
            return {
                'error': True,
                'message': 'No prediction context set. Please make a prediction first.'
            }
        
        try:
            # Build context-aware prompt
            context_summary = self._build_chat_context()
            
            # Add user message to history
            self.chat_history.append({
                'role': 'user',
                'content': user_message
            })
            
            # Build messages for API call
            messages = [
                {
                    'role': 'system',
                    'content': self._get_chat_system_prompt()
                },
                {
                    'role': 'assistant',
                    'content': f"I'm analyzing this detection:\n{context_summary}"
                }
            ] + self.chat_history
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=800
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.chat_history.append({
                'role': 'assistant',
                'content': assistant_message
            })
            
            return {
                'error': False,
                'message': assistant_message,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f"Chat error: {str(e)}"
            }
    
    def _get_chat_system_prompt(self):
        """
        System prompt for chat mode
        """
        return """You are a cybersecurity expert assistant helping a security analyst understand network anomaly detection results.

CONTEXT: You have access to a dual-model detection system (XGBoost + LSTM) with SHAP explainability. The analyst can ask you questions about:
- Why a specific prediction was made
- What features contributed most
- Whether the detection is reliable
- What actions to take
- How the models work
- Differences between model predictions

GUIDELINES:
1. Be concise and specific - the analyst is busy
2. Reference specific evidence from SHAP values and feature importance
3. Explain technical concepts clearly but don't oversimplify
4. If models disagree, explain both perspectives
5. Suggest concrete next steps when asked
6. Admit uncertainty when data is inconclusive
7. Use the actual values and SHAP scores from the detection

TONE: Professional, helpful, evidence-based. You're a colleague, not a chatbot."""
    
    def _build_chat_context(self):
        """
        Build a concise summary of current prediction for chat context
        """
        result = self.current_prediction_context
        ens = result['ensemble_decision']
        agents = result['agent_predictions']
        evidence = result['evidence']
        
        context = f"""Detection: {ens['final_prediction']} ({ens['confidence']*100:.1f}% confidence)
Agreement: {ens['agreement']}
XGBoost: {agents['xgboost']['prediction']} ({agents['xgboost']['confidence']*100:.1f}%)
LSTM: {agents['lstm']['prediction']} ({agents['lstm']['confidence']*100:.1f}%)

Top XGBoost Features:
"""
        for i, feat in enumerate(evidence['xgboost_evidence']['top_5_features'][:3], 1):
            context += f"{i}. {feat['feature']}: {feat['value']:.4f} (SHAP: {feat['shap_impact']:.4f})\n"
        
        context += "\nTop LSTM Features:\n"
        for i, feat in enumerate(evidence['lstm_evidence']['top_5_features'][:3], 1):
            context += f"{i}. {feat['feature']}: {feat['value']:.4f} (SHAP: {feat['shap_impact']:.4f})\n"
        
        return context
    
    def clear_chat_history(self):
        """
        Clear chat history (called when new prediction is made)
        """
        self.chat_history = []
        self.current_prediction_context = None

def main():
    """
    Main execution - DO NOT run without API key from user
    """
    print("\n" + "=" * 80)
    print("LLM AGENT - NATURAL LANGUAGE EXPLANATION GENERATOR")
    print("=" * 80)
    print("\n⚠️  This module requires OpenAI API key.")
    print("   Run from main pipeline: python mas_anomaly_detection.py")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

