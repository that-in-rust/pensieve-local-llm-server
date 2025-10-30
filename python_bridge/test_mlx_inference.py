#!/usr/bin/env python3
"""
Test Suite for MLX Inference Bridge - Memory Safety Validation

Following TDD Principles (S01):
- Executable Specifications: Tests define contract
- Write tests FIRST (RED → GREEN → REFACTOR)
- Validate all memory safety claims

Test Coverage:
1. Memory status checking (5 tests)
2. Cache clearing functionality (2 tests)
3. Memory logging (1 test)
4. Pre-inference memory checks (3 tests)
5. Post-generation cache clearing (2 tests)

Total: 13 tests for complete memory safety validation
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
from io import StringIO

# Import the module under test
import mlx_inference


class TestMemoryStatusChecking(unittest.TestCase):
    """Test memory status detection (Executable Specification #1)"""

    @patch('mlx_inference.psutil')
    def test_memory_status_safe_above_2gb(self, mock_psutil):
        """
        GIVEN: System has 4GB available RAM
        WHEN: check_memory_status() is called
        THEN: Returns ('SAFE', 4.0)
        """
        # Arrange
        mock_mem = Mock()
        mock_mem.available = 4 * (1024 ** 3)  # 4GB in bytes
        mock_psutil.virtual_memory.return_value = mock_mem
        mlx_inference.PSUTIL_AVAILABLE = True

        # Act
        status, available_gb = mlx_inference.check_memory_status()

        # Assert
        self.assertEqual(status, 'SAFE')
        self.assertAlmostEqual(available_gb, 4.0, places=2)

    @patch('mlx_inference.psutil')
    def test_memory_status_warning_between_1gb_and_2gb(self, mock_psutil):
        """
        GIVEN: System has 1.5GB available RAM
        WHEN: check_memory_status() is called
        THEN: Returns ('WARNING', 1.5)
        """
        # Arrange
        mock_mem = Mock()
        mock_mem.available = 1.5 * (1024 ** 3)  # 1.5GB
        mock_psutil.virtual_memory.return_value = mock_mem
        mlx_inference.PSUTIL_AVAILABLE = True

        # Act
        status, available_gb = mlx_inference.check_memory_status()

        # Assert
        self.assertEqual(status, 'WARNING')
        self.assertAlmostEqual(available_gb, 1.5, places=2)

    @patch('mlx_inference.psutil')
    def test_memory_status_critical_between_500mb_and_1gb(self, mock_psutil):
        """
        GIVEN: System has 0.75GB (750MB) available RAM
        WHEN: check_memory_status() is called
        THEN: Returns ('CRITICAL', 0.75)

        This is the threshold where requests are rejected (D17 spec)
        """
        # Arrange
        mock_mem = Mock()
        mock_mem.available = 0.75 * (1024 ** 3)  # 750MB
        mock_psutil.virtual_memory.return_value = mock_mem
        mlx_inference.PSUTIL_AVAILABLE = True

        # Act
        status, available_gb = mlx_inference.check_memory_status()

        # Assert
        self.assertEqual(status, 'CRITICAL')
        self.assertAlmostEqual(available_gb, 0.75, places=2)

    @patch('mlx_inference.psutil')
    def test_memory_status_emergency_below_500mb(self, mock_psutil):
        """
        GIVEN: System has 0.3GB (300MB) available RAM
        WHEN: check_memory_status() is called
        THEN: Returns ('EMERGENCY', 0.3)

        This triggers emergency shutdown (D17 spec)
        """
        # Arrange
        mock_mem = Mock()
        mock_mem.available = 0.3 * (1024 ** 3)  # 300MB
        mock_psutil.virtual_memory.return_value = mock_mem
        mlx_inference.PSUTIL_AVAILABLE = True

        # Act
        status, available_gb = mlx_inference.check_memory_status()

        # Assert
        self.assertEqual(status, 'EMERGENCY')
        self.assertAlmostEqual(available_gb, 0.3, places=2)

    def test_memory_status_unknown_when_psutil_unavailable(self):
        """
        GIVEN: psutil is not available
        WHEN: check_memory_status() is called
        THEN: Returns ('UNKNOWN', 0.0) gracefully
        """
        # Arrange
        original_state = mlx_inference.PSUTIL_AVAILABLE
        mlx_inference.PSUTIL_AVAILABLE = False

        try:
            # Act
            status, available_gb = mlx_inference.check_memory_status()

            # Assert
            self.assertEqual(status, 'UNKNOWN')
            self.assertEqual(available_gb, 0.0)
        finally:
            # Restore
            mlx_inference.PSUTIL_AVAILABLE = original_state


class TestCacheClearingFunctionality(unittest.TestCase):
    """Test MLX cache clearing (Executable Specification #2)"""

    @patch('mlx_inference.mx')
    @patch('sys.stderr', new_callable=StringIO)
    def test_clear_mlx_cache_when_available(self, mock_stderr, mock_mx):
        """
        GIVEN: MLX has metal.clear_cache() method
        WHEN: clear_mlx_cache() is called
        THEN: mx.metal.clear_cache() is invoked
        AND: Logs "[MEMORY] MLX cache cleared"
        """
        # Arrange
        mock_mx.metal.clear_cache = Mock()

        # Act
        mlx_inference.clear_mlx_cache()

        # Assert
        mock_mx.metal.clear_cache.assert_called_once()
        self.assertIn("[MEMORY] MLX cache cleared", mock_stderr.getvalue())

    @patch('mlx_inference.mx')
    @patch('sys.stderr', new_callable=StringIO)
    def test_clear_mlx_cache_handles_missing_method(self, mock_stderr, mock_mx):
        """
        GIVEN: MLX does not have metal.clear_cache() method
        WHEN: clear_mlx_cache() is called
        THEN: Does not crash
        AND: Logs warning about unavailable method
        """
        # Arrange
        del mock_mx.metal.clear_cache  # Simulate missing method

        # Act
        mlx_inference.clear_mlx_cache()

        # Assert
        self.assertIn("[WARN]", mock_stderr.getvalue())


class TestMemoryLogging(unittest.TestCase):
    """Test memory state logging (Executable Specification #3)"""

    @patch('mlx_inference.psutil')
    @patch('sys.stderr', new_callable=StringIO)
    def test_log_memory_state_includes_available_and_total(self, mock_stderr, mock_psutil):
        """
        GIVEN: System has 12GB total, 8GB available
        WHEN: log_memory_state("TEST") is called
        THEN: Logs memory information with label
        AND: Includes available/total GB and percentage
        """
        # Arrange
        mock_mem = Mock()
        mock_mem.available = 8 * (1024 ** 3)  # 8GB
        mock_mem.total = 12 * (1024 ** 3)     # 12GB
        mock_mem.percent = 33.3
        mock_psutil.virtual_memory.return_value = mock_mem
        mlx_inference.PSUTIL_AVAILABLE = True

        # Act
        mlx_inference.log_memory_state("TEST")

        # Assert
        output = mock_stderr.getvalue()
        self.assertIn("[MEMORY TEST]", output)
        self.assertIn("8.00GB", output)
        self.assertIn("12.00GB", output)
        self.assertIn("33.3%", output)


class TestPreInferenceMemoryChecks(unittest.TestCase):
    """Test memory checks before inference (Executable Specification #4)"""

    @patch('mlx_inference.check_memory_status')
    @patch('mlx_inference.log_memory_state')
    @patch('mlx_inference.clear_mlx_cache')
    @patch('mlx_inference.optimize_mlx_performance')
    @patch('mlx_inference.mlx_generate')
    def test_generation_rejects_at_critical_memory(
        self, mock_generate, mock_optimize, mock_clear, mock_log, mock_check
    ):
        """
        GIVEN: System memory is CRITICAL (0.8GB available)
        WHEN: real_mlx_generate() is called
        THEN: Raises RuntimeError before generation
        AND: Error message includes "Critical memory pressure"

        This is the PRIMARY SAFETY GUARANTEE from D17
        """
        # Arrange
        mock_check.return_value = ('CRITICAL', 0.8)
        model_data = {'model': Mock(), 'tokenizer': Mock()}

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            list(mlx_inference.real_mlx_generate(
                model_data, "test prompt", max_tokens=10
            ))

        self.assertIn("Critical memory pressure", str(context.exception))
        self.assertIn("0.80GB", str(context.exception))
        mock_generate.assert_not_called()  # Should NOT generate

    @patch('mlx_inference.check_memory_status')
    @patch('mlx_inference.log_memory_state')
    @patch('mlx_inference.clear_mlx_cache')
    @patch('mlx_inference.optimize_mlx_performance')
    @patch('mlx_inference.mlx_generate')
    def test_generation_rejects_at_emergency_memory(
        self, mock_generate, mock_optimize, mock_clear, mock_log, mock_check
    ):
        """
        GIVEN: System memory is EMERGENCY (0.3GB available)
        WHEN: real_mlx_generate() is called
        THEN: Raises RuntimeError immediately
        AND: Clears MLX cache before shutdown
        AND: Error message includes "Emergency"
        """
        # Arrange
        mock_check.return_value = ('EMERGENCY', 0.3)
        model_data = {'model': Mock(), 'tokenizer': Mock()}

        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            list(mlx_inference.real_mlx_generate(
                model_data, "test prompt", max_tokens=10
            ))

        self.assertIn("Emergency", str(context.exception))
        mock_clear.assert_called()  # Should clear cache
        mock_generate.assert_not_called()

    @patch('mlx_inference.check_memory_status')
    @patch('mlx_inference.log_memory_state')
    @patch('mlx_inference.clear_mlx_cache')
    @patch('mlx_inference.optimize_mlx_performance')
    @patch('mlx_inference.mlx_generate')
    def test_generation_proceeds_with_safe_memory(
        self, mock_generate, mock_optimize, mock_clear, mock_log, mock_check
    ):
        """
        GIVEN: System memory is SAFE (4GB available)
        WHEN: real_mlx_generate() is called
        THEN: Proceeds with generation normally
        AND: Calls mlx_generate()
        """
        # Arrange
        mock_check.return_value = ('SAFE', 4.0)
        mock_generate.return_value = "Generated response"
        model_data = {'model': Mock(), 'tokenizer': Mock()}

        # Act
        result = list(mlx_inference.real_mlx_generate(
            model_data, "test prompt", max_tokens=10, stream=False
        ))

        # Assert
        mock_generate.assert_called_once()
        self.assertEqual(result[0], "Generated response")


class TestPostGenerationCacheClearing(unittest.TestCase):
    """Test cache clearing after generation (Executable Specification #5)"""

    @patch('mlx_inference.check_memory_status')
    @patch('mlx_inference.log_memory_state')
    @patch('mlx_inference.clear_mlx_cache')
    @patch('mlx_inference.optimize_mlx_performance')
    @patch('mlx_inference.mlx_generate')
    def test_cache_cleared_after_successful_generation(
        self, mock_generate, mock_optimize, mock_clear, mock_log, mock_check
    ):
        """
        GIVEN: Generation completes successfully
        WHEN: real_mlx_generate() finishes
        THEN: clear_mlx_cache() is called
        AND: log_memory_state("AFTER") is called

        This prevents MLX memory leaks (issues #724, #1124)
        """
        # Arrange
        mock_check.return_value = ('SAFE', 4.0)
        mock_generate.return_value = "Response"
        model_data = {'model': Mock(), 'tokenizer': Mock()}

        # Act
        list(mlx_inference.real_mlx_generate(
            model_data, "prompt", max_tokens=10, stream=False
        ))

        # Assert
        mock_clear.assert_called()
        mock_log.assert_any_call("AFTER")

    @patch('mlx_inference.check_memory_status')
    @patch('mlx_inference.log_memory_state')
    @patch('mlx_inference.clear_mlx_cache')
    @patch('mlx_inference.optimize_mlx_performance')
    @patch('mlx_inference.mlx_generate')
    def test_cache_cleared_even_on_generation_error(
        self, mock_generate, mock_optimize, mock_clear, mock_log, mock_check
    ):
        """
        GIVEN: Generation raises an exception
        WHEN: real_mlx_generate() encounters error
        THEN: clear_mlx_cache() is STILL called (finally block)

        Critical: Must clear cache even on errors to prevent leaks
        """
        # Arrange
        mock_check.return_value = ('SAFE', 4.0)
        mock_generate.side_effect = RuntimeError("Generation failed")
        model_data = {'model': Mock(), 'tokenizer': Mock()}

        # Act
        try:
            list(mlx_inference.real_mlx_generate(
                model_data, "prompt", max_tokens=10, stream=False
            ))
        except RuntimeError:
            pass  # Expected

        # Assert
        mock_clear.assert_called()  # Must be called even on error


class TestMemoryThresholds(unittest.TestCase):
    """Test memory threshold constants (Executable Specification #6)"""

    def test_critical_threshold_is_1gb(self):
        """
        GIVEN: D17 research specifies 1GB critical threshold
        WHEN: Checking MEMORY_CRITICAL_GB constant
        THEN: Value equals 1.0
        """
        self.assertEqual(mlx_inference.MEMORY_CRITICAL_GB, 1.0)

    def test_emergency_threshold_is_500mb(self):
        """
        GIVEN: D17 research specifies 500MB emergency threshold
        WHEN: Checking MEMORY_EMERGENCY_GB constant
        THEN: Value equals 0.5
        """
        self.assertEqual(mlx_inference.MEMORY_EMERGENCY_GB, 0.5)


def run_tests_with_coverage():
    """Run tests and show coverage report"""
    print("=" * 70)
    print("MLX Inference Bridge - Memory Safety Test Suite")
    print("Following S01 TDD Principles: RED → GREEN → REFACTOR")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryStatusChecking))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheClearingFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestPreInferenceMemoryChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestPostGenerationCacheClearing))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryThresholds))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()

    if result.wasSuccessful():
        print("✅ ALL TESTS PASSING - Memory safety validated")
        return 0
    else:
        print("❌ TESTS FAILING - Implementation needs fixes")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests_with_coverage())
