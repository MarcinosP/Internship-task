from rest_framework import serializers
from .models import Covertype

class CovertypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Covertype
        fields = '__all__'