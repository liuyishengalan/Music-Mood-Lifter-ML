from django.db import transaction
from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from rest_framework import mixins
from rest_framework.exceptions import APIException

from .models import MLAlgorithm, MLRequest
from . import models
from . import serializers

import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
from ml.registry import MLRegistry
from mysite.wsgi import registry


class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = serializers.EndpointSerializer
    queryset = models.Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = serializers.MLAlgorithmSerializer
    queryset = models.MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = models.MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
                                                           created_at__lt=instance.created_at,
                                                           active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    models.MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = serializers.MLAlgorithmStatusSerializer
    queryset = models.MLAlgorithmStatus.objects.all()

    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)

        except Exception as e:
            raise APIException(str(e))


class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = serializers.MLRequestSerializer
    queryset = models.MLRequest.objects.all()


class PredictView(views.APIView):
    def post(self, request, endpoint_name, format=None):

        algorithm_status = self.request.query_params.get("status", "production")
        algorithm_version = self.request.query_params.get("version")

        algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
                                          status__active=True)

        if algorithm_version is not None:
            algs = algs.filter(version=algorithm_version)

        if len(algs) == 0:
            return Response(
                {"status": "Error", "message": "ML algorithm is not available"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if len(algs) != 1 and algorithm_status != "ab_testing":
            return Response(
                {"status": "Error",
                 "message": "ML algorithm selection is ambiguous. Please specify algorithm version."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alg_index = 0
        if algorithm_status == "ab_testing":
            alg_index = 0 if rand() < 0.5 else 1

        algorithm_object = registry.endpoints[algs[alg_index].id]
        prediction = algorithm_object.getTypicalTracks(request.data)

        ml_request = MLRequest(
            input_data=int(request.data),
            full_response=prediction,
            response=prediction,
            feedback="",
            parent_mlalgorithm=algs[alg_index],
        )
        ml_request.save()

        #prediction["request_id"] = ml_request.id

        return Response(prediction)
