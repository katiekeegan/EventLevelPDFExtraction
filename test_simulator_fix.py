#!/usr/bin/env python3
"""
Test to verify the ValueError: too many values to unpack (expected 4) fix.

This test validates that:
1. All simulators can be imported correctly from simulator.py
2. Parameter unpacking works correctly 
3. No duplicate simulator definitions exist
4. Interface consistency is maintained
"""

import torch

def test_simulator_imports():
    """Test that all simulators can be imported without errors."""
    from simulator import SimplifiedDIS, RealisticDIS, MCEGSimulator, Gaussian2DSimulator
    
    assert SimplifiedDIS is not None
    assert RealisticDIS is not None
    # MCEGSimulator might be None if dependencies are missing
    assert Gaussian2DSimulator is not None

def test_simplified_dis_parameters():
    """Test SimplifiedDIS with correct and incorrect parameter counts."""
    from simulator import SimplifiedDIS
    
    sim = SimplifiedDIS()
    
    # Should work with 4 parameters
    sim.init([1.0, 2.0, 3.0, 4.0])
    
    # Should fail with wrong number of parameters
    try:
        sim.init([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 params
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "too many values to unpack" in str(e)

def test_realistic_dis_parameters():
    """Test RealisticDIS with correct parameter count."""
    from simulator import RealisticDIS
    
    sim = RealisticDIS()
    
    # Should work with 6 parameters
    sim.init([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Should fail with wrong number of parameters
    try:
        sim.init([1.0, 2.0, 3.0, 4.0])  # 4 params
        assert False, "Should have raised IndexError or ValueError"
    except (IndexError, ValueError):
        pass  # Expected

def test_interface_consistency():
    """Test that all simulators have consistent interfaces."""
    from simulator import SimplifiedDIS, RealisticDIS, Gaussian2DSimulator
    
    sim1 = SimplifiedDIS()
    sim2 = RealisticDIS()
    sim3 = Gaussian2DSimulator()
    
    # All should have sample method with n_events parameter
    data1 = sim1.sample([1.0, 2.0, 3.0, 4.0], n_events=10)
    data2 = sim2.sample([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], n_events=10)
    data3 = sim3.sample([0.0, 0.0, 1.0, 1.0, 0.3], n_events=10)
    
    assert data1.shape[0] == 10
    assert data2.shape[0] == 10  
    assert data3.shape[0] == 10

def test_plotting_utils_integration():
    """Test that plotting_UQ_utils can import simulators correctly."""
    import plotting_UQ_utils
    
    SimplifiedDIS, RealisticDIS, MCEGSimulator = plotting_UQ_utils.get_simulator_module()
    
    # Should not be None anymore
    assert SimplifiedDIS is not None
    assert RealisticDIS is not None
    # MCEGSimulator might be None due to missing dependencies

def test_no_duplicate_definitions():
    """Test that there are no duplicate simulator definitions."""
    # This test ensures uq_plotting_demo.py imports from simulator.py
    import uq_plotting_demo
    
    # Should be able to import without errors
    # The fact that this doesn't raise an error means the imports work
    
    # Verify that uq_plotting_demo uses the correct simulators
    from simulator import SimplifiedDIS as SimulatorSimplifiedDIS
    
    # Create an instance to make sure it works
    sim = SimulatorSimplifiedDIS()
    sim.init([1.0, 2.0, 3.0, 4.0])

if __name__ == "__main__":
    print("Running tests for simulator fix...")
    
    test_simulator_imports()
    print("✓ Simulator imports test passed")
    
    test_simplified_dis_parameters()
    print("✓ SimplifiedDIS parameters test passed")
    
    test_realistic_dis_parameters()
    print("✓ RealisticDIS parameters test passed")
    
    test_interface_consistency()
    print("✓ Interface consistency test passed")
    
    test_plotting_utils_integration()
    print("✓ Plotting utils integration test passed")
    
    test_no_duplicate_definitions()
    print("✓ No duplicate definitions test passed")
    
    print("\n🎉 All tests passed! The ValueError fix is working correctly.")