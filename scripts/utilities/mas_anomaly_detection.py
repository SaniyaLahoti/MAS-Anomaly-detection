"""
Multi-Agent System for Network Anomaly Detection with Explainability

Complete pipeline integrating:
1. XGBoost Agent (hierarchical classification + SHAP)
2. LSTM Agent (hierarchical classification + SHAP)
3. Interpreter Agent (ensemble voting + evidence combination)
4. LLM Agent (natural language explanation generation)

Output Format: Alert → Evidence → Impact → Recommendation
"""

import pandas as pd
import numpy as np
import json
from interpreter_agent import InterpreterAgent
from llm_agent import LLMAgent
from datetime import datetime
import argparse
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

class MultiAgentAnomalyDetectionSystem:
    """
    Complete Multi-Agent System orchestrator
    """
    
    def __init__(self, openai_api_key):
        """
        Initialize all agents
        """
        self.openai_api_key = openai_api_key
        self.interpreter_agent = None
        self.llm_agent = None
        
    def initialize_system(self):
        """
        Initialize all agents in the system
        """
        print("\n" + "=" * 80)
        print("MULTI-AGENT ANOMALY DETECTION SYSTEM - INITIALIZATION")
        print("=" * 80)
        print("\nSystem Components:")
        print("  🤖 Agent 1: XGBoost Hierarchical Classifier + SHAP")
        print("  🤖 Agent 2: LSTM Hierarchical Classifier + SHAP")
        print("  🔮 Agent 3: Interpreter (Ensemble Combiner)")
        print("  🧠 Agent 4: LLM (Natural Language Generator)")
        print("=" * 80)
        
        # Initialize Interpreter Agent (includes XGBoost + LSTM agents)
        print("\n📦 Initializing Interpreter Agent...")
        self.interpreter_agent = InterpreterAgent()
        df = self.interpreter_agent.initialize_agents()
        print("✅ Interpreter Agent ready (XGBoost + LSTM loaded)")
        
        # Initialize LLM Agent
        print("\n📦 Initializing LLM Agent...")
        self.llm_agent = LLMAgent(api_key=self.openai_api_key)
        print("✅ LLM Agent ready (OpenAI connection established)")
        
        print("\n" + "=" * 80)
        print("✅ MULTI-AGENT SYSTEM READY")
        print("=" * 80)
        
        return df
    
    def detect_and_explain(self, sample_data, actual_label=None, 
                          source_ip="Unknown", affected_assets="System"):
        """
        Single sample detection and explanation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 80)
        print(f"MULTI-AGENT DETECTION - {timestamp}")
        print("=" * 80)
        
        # Step 1: Ensemble prediction
        print("\n🔮 Step 1: Ensemble Prediction (XGBoost + LSTM)")
        ensemble_result = self.interpreter_agent.ensemble_predict(
            sample_data,
            actual_label=actual_label,
            timestamp=timestamp,
            source_ip=source_ip,
            affected_assets=affected_assets
        )
        
        # Step 2: LLM explanation
        print("\n🧠 Step 2: Generating Natural Language Explanation (LLM)")
        llm_explanation = self.llm_agent.generate_explanation(ensemble_result)
        
        # Combine
        result = {
            'timestamp': timestamp,
            'ensemble_result': ensemble_result,
            'llm_explanation': llm_explanation,
            'actual_label': actual_label
        }
        
        return result
    
    def process_batch(self, df_samples):
        """
        Process batch of samples through complete pipeline
        """
        print("\n" + "=" * 80)
        print("BATCH PROCESSING - MULTI-AGENT PIPELINE")
        print("=" * 80)
        print(f"Processing {len(df_samples)} samples...")
        
        # Step 1: Ensemble predictions for all samples
        print("\n" + "=" * 80)
        print("STEP 1: ENSEMBLE PREDICTIONS")
        print("=" * 80)
        ensemble_results = self.interpreter_agent.process_multiple_samples(df_samples)
        
        # Save intermediate results
        self.interpreter_agent.save_ensemble_results(
            ensemble_results, 
            filename='batch_ensemble_results.json'
        )
        
        # Step 2: LLM explanations for all results
        print("\n" + "=" * 80)
        print("STEP 2: LLM EXPLANATIONS")
        print("=" * 80)
        llm_results = self.llm_agent.process_ensemble_results(ensemble_results)
        
        # Save LLM results
        self.llm_agent.save_results(llm_results, filename='batch_llm_explanations.json')
        self.llm_agent.save_readable_report(llm_results, filename='batch_security_report.txt')
        
        # Print first sample
        if llm_results:
            print("\n" + "=" * 80)
            print("SAMPLE ALERT (First Detection)")
            print("=" * 80)
            print(llm_results[0]['llm_explanation'])
            print("=" * 80)
        
        return llm_results
    
    def generate_system_report(self, llm_results):
        """
        Generate comprehensive system performance report
        """
        print("\n" + "=" * 80)
        print("GENERATING SYSTEM PERFORMANCE REPORT")
        print("=" * 80)
        
        total = len(llm_results)
        
        # Accuracy metrics
        correct = sum(1 for r in llm_results if r['ensemble_result']['correct'])
        accuracy = correct / total * 100 if total > 0 else 0
        
        # Agreement metrics
        agreements = sum(
            1 for r in llm_results 
            if r['ensemble_result']['ensemble_decision']['agreement'] == 'FULL_AGREEMENT'
        )
        agreement_rate = agreements / total * 100 if total > 0 else 0
        
        # Confidence metrics
        avg_confidence = np.mean([
            r['ensemble_result']['ensemble_decision']['confidence'] 
            for r in llm_results
        ]) * 100
        
        # Per-class breakdown
        predictions_by_class = {}
        correct_by_class = {}
        
        for r in llm_results:
            pred = r['ensemble_result']['ensemble_decision']['final_prediction']
            actual = r['ensemble_result']['actual_label']
            
            if pred not in predictions_by_class:
                predictions_by_class[pred] = 0
                correct_by_class[pred] = 0
            
            predictions_by_class[pred] += 1
            if pred == actual:
                correct_by_class[pred] += 1
        
        # Generate report
        report = {
            'system_metrics': {
                'total_samples': total,
                'overall_accuracy': f"{accuracy:.2f}%",
                'model_agreement_rate': f"{agreement_rate:.2f}%",
                'average_confidence': f"{avg_confidence:.2f}%",
                'correct_predictions': f"{correct}/{total}"
            },
            'per_class_performance': {},
            'model_comparison': {
                'xgboost_wins': sum(
                    1 for r in llm_results
                    if r['ensemble_result']['ensemble_decision']['agreement'] == 'DISAGREEMENT'
                    and r['ensemble_result']['agent_predictions']['xgboost']['prediction'] == 
                        r['ensemble_result']['ensemble_decision']['final_prediction']
                ),
                'lstm_wins': sum(
                    1 for r in llm_results
                    if r['ensemble_result']['ensemble_decision']['agreement'] == 'DISAGREEMENT'
                    and r['ensemble_result']['agent_predictions']['lstm']['prediction'] == 
                        r['ensemble_result']['ensemble_decision']['final_prediction']
                ),
                'full_agreements': agreements
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for cls in predictions_by_class:
            total_cls = predictions_by_class[cls]
            correct_cls = correct_by_class[cls]
            accuracy_cls = correct_cls / total_cls * 100 if total_cls > 0 else 0
            
            report['per_class_performance'][cls] = {
                'predictions': total_cls,
                'correct': correct_cls,
                'accuracy': f"{accuracy_cls:.2f}%"
            }
        
        # Save report
        with open('mas_system_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("MULTI-AGENT SYSTEM PERFORMANCE REPORT")
        print("=" * 80)
        print(f"\n📊 OVERALL METRICS:")
        print(f"   Total Samples:      {total}")
        print(f"   Accuracy:           {accuracy:.2f}%")
        print(f"   Agreement Rate:     {agreement_rate:.2f}%")
        print(f"   Avg Confidence:     {avg_confidence:.2f}%")
        
        print(f"\n🤖 MODEL COMPARISON:")
        print(f"   Full Agreements:    {report['model_comparison']['full_agreements']}")
        print(f"   XGBoost Wins:       {report['model_comparison']['xgboost_wins']}")
        print(f"   LSTM Wins:          {report['model_comparison']['lstm_wins']}")
        
        print(f"\n📈 PER-CLASS PERFORMANCE:")
        for cls, metrics in report['per_class_performance'].items():
            print(f"   {cls:<20} Accuracy: {metrics['accuracy']:<8} ({metrics['correct']}/{metrics['predictions']})")
        
        print("\n✅ Full report saved to 'mas_system_report.json'")
        print("=" * 80)
        
        return report

def main():
    """
    Main execution
    """
    parser = argparse.ArgumentParser(
        description='Multi-Agent Network Anomaly Detection System'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key',
        default=None
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=15,
        help='Number of test samples to process (default: 15)'
    )
    
    args = parser.parse_args()
    
    # Get API key from command line arg, environment variable, or .env file
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("ERROR: OpenAI API key required!")
        print("Please either:")
        print("  1. Set OPENAI_API_KEY in your .env file")
        print("  2. Set OPENAI_API_KEY environment variable")
        print("  3. Use --api-key YOUR_KEY command line argument")
        return
    
    print("\n" + "=" * 80)
    print("MULTI-AGENT NETWORK ANOMALY DETECTION SYSTEM")
    print("WITH EXPLAINABLE AI (SHAP + LLM)")
    print("=" * 80)
    print("\n🤖 Agents:")
    print("   1. XGBoost Hierarchical Classifier + SHAP")
    print("   2. LSTM Hierarchical Classifier + SHAP")
    print("   3. Interpreter Agent (Ensemble)")
    print("   4. LLM Agent (Natural Language)")
    print("\n📊 Output Format:")
    print("   Alert → Evidence → Impact → Recommendation")
    print("=" * 80)
    
    # Initialize system
    system = MultiAgentAnomalyDetectionSystem(openai_api_key=api_key)
    df = system.initialize_system()
    
    # Select test samples
    print(f"\n📊 Selecting {args.num_samples} test samples...")
    test_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(3, len(x)), random_state=789)
    ).reset_index(drop=True)
    
    if len(test_samples) > args.num_samples:
        test_samples = test_samples.sample(n=args.num_samples, random_state=42)
    
    print(f"✅ Selected {len(test_samples)} samples")
    print(f"\nDistribution:")
    print(test_samples['Attack'].value_counts())
    
    # Process batch
    llm_results = system.process_batch(test_samples)
    
    # Generate system report
    system_report = system.generate_system_report(llm_results)
    
    print("\n" + "=" * 80)
    print("✅ MULTI-AGENT SYSTEM EXECUTION COMPLETE")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("   • batch_ensemble_results.json     - Ensemble predictions")
    print("   • batch_llm_explanations.json     - LLM explanations (JSON)")
    print("   • batch_security_report.txt       - Human-readable alerts")
    print("   • mas_system_report.json          - System performance")
    print("\n🎯 Use cases:")
    print("   • Production deployment: Run on live network traffic")
    print("   • Research: Analyze model disagreements and ensemble benefits")
    print("   • Security: Review alerts in batch_security_report.txt")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

