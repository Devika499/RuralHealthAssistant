#!/usr/bin/env python3
"""
Test script for RLHF module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rlhf_module import rlhf

def test_rlhf():
    """Test RLHF module functionality"""
    print("Testing RLHF Module...")
    
    # Test 1: Check if reward model can be loaded
    print("\n1. Testing reward model loading...")
    success = rlhf.load_reward_model()
    if success:
        print("✅ Reward model loaded successfully")
    else:
        print("❌ Failed to load reward model")
        return False
    
    # Test 2: Check model status
    print("\n2. Testing model status...")
    status = rlhf.get_model_status()
    print(f"Status: {status}")
    
    # Test 3: Test response scoring
    print("\n3. Testing response scoring...")
    test_prompt = "I have a headache and fever"
    test_response = "Based on your symptoms, you might have a common cold or flu. Rest well and stay hydrated."
    
    score_result = rlhf.score_response(test_prompt, test_response)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {test_response}")
    print(f"Score Result: {score_result}")
    
    # Test 4: Test with different response quality
    print("\n4. Testing with different response quality...")
    poor_response = "I don't know what to say about that."
    good_response = "Your symptoms suggest you may have a viral infection. I recommend: 1) Rest and hydration, 2) Monitor your temperature, 3) Contact a doctor if symptoms worsen."
    
    poor_score = rlhf.score_response(test_prompt, poor_response)
    good_score = rlhf.score_response(test_prompt, good_response)
    
    print(f"Poor Response: {poor_response}")
    print(f"Poor Score: {poor_score}")
    print(f"Good Response: {good_response}")
    print(f"Good Score: {good_score}")
    
    print("\n✅ RLHF module test completed!")
    return True

if __name__ == "__main__":
    test_rlhf() 