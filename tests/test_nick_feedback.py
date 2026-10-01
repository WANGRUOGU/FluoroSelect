import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock
from streamlit.testing.v1 import AppTest
from pathlib import Path
import ai_helper
from tutorial import merge_probe_options


class NickFeedbackTests(unittest.TestCase):
    def test_purchase_replaces_alias_options_without_changing_inventory(self):
        original = {'EUB 338': ['A'], 'other': ['B']}
        result = merge_probe_options(original, {'EUB338': ['C'], 'new': ['B', 'C']}, {'A': {}, 'B': {}, 'C': {}})
        self.assertEqual(result['EUB 338'], ['C'])
        self.assertEqual(result['new'], ['B', 'C'])
        self.assertEqual(original['EUB 338'], ['A'])

    def test_unknown_purchase_dye_rejected(self):
        with self.assertRaises(ValueError):
            merge_probe_options({}, {'new': ['unknown']}, {'A': {}})

    def test_retry_503_then_success(self):
        error = RuntimeError('private provider payload')
        error.code = 503
        client = Mock()
        client.models.generate_content.side_effect = [error, SimpleNamespace(text='ok')]
        with patch.object(ai_helper, 'get_gemini_client', return_value=client), patch.object(ai_helper, 'get_model_name', return_value='test'), patch.object(ai_helper.time, 'sleep') as sleep:
            self.assertEqual(ai_helper.call_gemini('request'), 'ok')
            self.assertEqual(sleep.call_count, 1)

    def test_retries_bounded_and_error_safe(self):
        error = RuntimeError('secret payload')
        error.code = 503
        client = Mock()
        client.models.generate_content.side_effect = error
        with patch.object(ai_helper, 'get_gemini_client', return_value=client), patch.object(ai_helper, 'get_model_name', return_value='test'), patch.object(ai_helper.time, 'sleep'):
            with self.assertRaisesRegex(ai_helper.AIServiceError, 'temporarily busy'):
                ai_helper.call_gemini('request')
        self.assertEqual(client.models.generate_content.call_count, 3)

    def test_sdk_retry_settings_valid(self):
        from google.genai import types
        options = types.HttpOptions(timeout=15000, retry_options=types.HttpRetryOptions(attempts=1))
        self.assertEqual(options.retry_options.attempts, 1)

    def test_tutorial_and_display_minimum(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.slider(key='k_show_slider').min, 1)
        app.radio[0].set_value('Tutorial').run()
        self.assertFalse(app.exception)
        self.assertIn('FluoroSelect tutorial', [x.value for x in app.header])

    def test_purchase_ui_and_ai_failure_keep_manual_controls(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run()
        app.text_input[0].set_value('Prospective probe')
        app.multiselect[0].set_value([app.multiselect[0].options[0]])
        next(b for b in app.button if b.label == 'Apply probe options').click().run()
        self.assertFalse(app.exception)
        self.assertIn('Prospective probe', app.multiselect(key='picked_additional_probes').options)
        with patch('ai_ui.parse_user_request', side_effect=ai_helper.AIServiceError('temporarily busy')):
            app.text_input(key='ai_main_input').set_value('Choose EUB338').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key='ai_main_input').value, 'Choose EUB338')
        self.assertTrue(any(b.label == 'Retry AI request' for b in app.button))
        self.assertEqual(app.radio(key='source_radio').value, 'By probes')
