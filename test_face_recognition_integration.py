"""
Test script for Face Recognition Integration
Tests the integration of face recognition with PPE detection system
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, Any

def test_face_recognition_engine():
    """Test the face recognition engine independently"""
    print("🧪 Testing Face Recognition Engine...")
    
    try:
        from face_recognition_engine import FaceRecognitionEngine
        
        # Initialize engine
        engine = FaceRecognitionEngine()
        print("✅ Face Recognition Engine initialized successfully")
        
        # Test dataset info
        dataset_info = engine.get_dataset_info()
        print(f"📊 Dataset Info: {dataset_info['total_people']} people, {dataset_info['total_samples']} samples")
        
        # Test face detection on a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        face_results = engine.detect_faces(dummy_image)
        print(f"👤 Face detection test: {len(face_results)} faces detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Face Recognition Engine test failed: {e}")
        return False

def test_ppe_detection_integration():
    """Test PPE detection engine with face recognition integration"""
    print("\n🧪 Testing PPE Detection Integration...")
    
    try:
        from ppe_detection_engine import PPEDetectionEngine
        
        # Initialize engine with face recognition
        engine = PPEDetectionEngine(enable_face_recognition=True)
        print("✅ PPE Detection Engine with Face Recognition initialized")
        
        # Check if face recognition is enabled
        if engine.is_face_recognition_enabled():
            print("✅ Face Recognition is enabled in PPE engine")
        else:
            print("⚠️ Face Recognition is disabled in PPE engine")
        
        # Test detection on dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = engine.detect_objects(dummy_image)
        
        if 'error' not in results:
            print("✅ PPE detection with face recognition works")
            print(f"📊 Detection results keys: {list(results.keys())}")
            
            # Check for face recognition results
            if 'face_results' in results:
                print(f"👤 Face results: {len(results['face_results'])} faces")
            if 'face_stats' in results:
                print(f"📈 Face stats: {results['face_stats']}")
        else:
            print(f"❌ PPE detection failed: {results['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ PPE Detection Integration test failed: {e}")
        return False

def test_webcam_component():
    """Test webcam component integration"""
    print("\n🧪 Testing Webcam Component...")
    
    try:
        from webcam_component import WebcamPPEDetector
        from ppe_detection_engine import PPEDetectionEngine
        
        # Initialize detection engine
        detection_engine = PPEDetectionEngine(enable_face_recognition=True)
        
        # Initialize webcam detector
        detector = WebcamPPEDetector(detection_engine)
        print("✅ Webcam component initialized with face recognition")
        
        # Check if face stats are initialized
        if hasattr(detector, 'face_stats'):
            print("✅ Face statistics tracking is available")
            print(f"📊 Face stats structure: {list(detector.face_stats.keys())}")
        else:
            print("⚠️ Face statistics tracking not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Webcam Component test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        'face_recognition_engine.py',
        'ppe_detection_engine.py',
        'webcam_component.py',
        'app_ultra_fast.py'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if all imports work correctly"""
    print("\n🧪 Testing Imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Face Recognition Integration Tests\n")
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Face Recognition Engine", test_face_recognition_engine),
        ("PPE Detection Integration", test_ppe_detection_integration),
        ("Webcam Component", test_webcam_component)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Face Recognition integration is ready!")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
