"""Tests for rainbow.utils."""

import logging
import os
import tempfile
import unittest

from rainbow import utils


class ConfigureLoggingTestCase(unittest.TestCase):
    """Test rainbow.utils.configure_logging."""

    def test_attaches_log_handler(self):
        n_handlers_before = len(logging.root.handlers)

        handler = utils.configure_logging()

        self.assertEqual(len(logging.root.handlers), n_handlers_before + 1)
        self.assertIn(handler, logging.root.handlers)

        logging.root.removeHandler(handler)

    def test_verbose_true_sets_log_level_to_debug(self):
        handler = utils.configure_logging(verbose=True)

        self.assertEqual(handler.level, logging.DEBUG)

        logging.root.removeHandler(handler)

    def test_verbose_false_sets_log_level_to_info(self):
        handler = utils.configure_logging(verbose=False)

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)

    def test_verbose_defaults_to_false(self):
        handler = utils.configure_logging()

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)


class FileLoggingTestCase(unittest.TestCase):
    """Test rainbow.utils.FileLogging."""

    def test_attaches_log_handler(self):
        n_handlers_before = len(logging.root.handlers)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.log")
            with utils.FileLogging(file_path) as handler:
                self.assertTrue(isinstance(handler, logging.Handler))
                self.assertEqual(
                    len(logging.root.handlers), n_handlers_before + 1
                )
                self.assertIn(handler, logging.root.handlers)

    def test_detaches_log_handler(self):
        n_handlers_before = len(logging.root.handlers)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.log")
            with utils.FileLogging(file_path) as handler:
                pass

        self.assertEqual(len(logging.root.handlers), n_handlers_before)
        self.assertNotIn(handler, logging.root.handlers)


class StringToSeedTestCase(unittest.TestCase):
    """Test rainbow.utils.string_to_seed."""

    def test_maps_a_string_to_an_int(self):
        self.assertIsInstance(utils.string_to_seed("foo"), int)

    def test_is_deterministic(self):
        self.assertEqual(utils.string_to_seed("foo"), 740734059)
