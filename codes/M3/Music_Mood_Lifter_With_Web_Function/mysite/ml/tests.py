from django.test import TestCase

from . import music_model
import inspect
from .registry import MLRegistry


class MLTests(TestCase):
    def test_music_algorithm(self):
        input_data = 2
        my_alg = music_model.MusicMoodClassifier()
        response = my_alg.getTypicalTracks(input_data)
        print(response)

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "music mood lifter"
        algorithm_object = music_model.MusicMoodClassifier()
        algorithm_name = "music_model"
        algorithm_status = "production"
        algorithm_version = "0.0.2"
        algorithm_owner = "team_torosaurus"
        algorithm_description = "music mood lifter"
        algorithm_code = inspect.getsource(music_model.MusicMoodClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)