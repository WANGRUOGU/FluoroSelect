import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class AppSmokeTests(unittest.TestCase):
    def assert_results(self, app):
        self.assertEqual([x.message for x in app.exception], [])
        self.assertEqual([x.value for x in app.error], [])
        self.assertEqual({x.label for x in app.metric}, {
            'Macro accuracy', 'Worst-class accuracy', 'True-class RMSE', 'Worst-class RMSE'})

    def test_pool_emission_and_predicted(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.radio(key='source_radio').set_value('EUB338 only').run()
        self.assert_results(app)
        app.radio(key='mode_radio').set_value('Predicted spectra').run()
        self.assert_results(app)

    def test_probe_assignment(self):
        app = AppTest.from_file(APP, default_timeout=90).run()
        app.multiselect(key='picked_additional_probes').set_value(['EUB338', 'ACT476', 'ALF968']).run()
        self.assert_results(app)


if __name__ == '__main__':
    unittest.main()
