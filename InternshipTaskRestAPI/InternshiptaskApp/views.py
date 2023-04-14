# task 6

from rest_framework import generics
from rest_framework.response import Response
from .serializers import CovertypeSerializer
import pickle
import pandas as pd
import numpy as np
from django.http import JsonResponse


class CoverTypePredictionView(generics.GenericAPIView):
    serializer_class = CovertypeSerializer

    def post(self, request):
        model_type = request.data.get('model_type')
        input_features = request.data.get('input_features')

        with open('models.pickle', 'rb') as f:
            data = pickle.load(f)

        logistic_regression_model = data['logistic_regression_model']
        decision_tree_model = data['decision_tree_model']
        tensorflow_model = data['tensorflow_model']

        if model_type == 'heuristic':
            prediction = self.heuristic_model()
        elif model_type == 'logistic_regression':
            prediction = logistic_regression_model.predict([list(input_features[0].values())])
        elif model_type == 'decision_tree':
            prediction = decision_tree_model.predict([list(input_features[0].values())])
        elif model_type == 'neural_network':
            prediction_nn = tensorflow_model.predict([list(input_features[0].values())])
            prediction = np.argmax(prediction_nn, axis=1)[0]
        else:
            return Response({'error': 'Invalid model type.'}, status=400)
        return_prediction = prediction.tolist()
        return JsonResponse({'model_type': model_type, 'prediction': return_prediction}, status=200)

    def heuristic_model(self):
            df = pd.read_csv('./InternshipTaskRestAPI/data/covtype.data', delimiter=',', header=None)
            return df.iloc[:, -1].value_counts().idxmax()
