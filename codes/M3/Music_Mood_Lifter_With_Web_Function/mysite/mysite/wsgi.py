"""
WSGI config for mysite project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

import inspect
from ml.registry import MLRegistry
from ml.music_model import MusicMoodClassifier

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

application = get_wsgi_application()

# ML registry


try:
    registry = MLRegistry()  # create ML registry
    # Random Forest classifier
    rf = MusicMoodClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="music_model",
                           algorithm_object=MusicMoodClassifier(),
                           algorithm_name="music_model",
                           algorithm_status="production",
                           algorithm_version="0.0.2",
                           owner="team_torosaurus",
                           algorithm_description="music mood lifter",
                           algorithm_code=inspect.getsource(MusicMoodClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
