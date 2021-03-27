from rest_framework import serializers

from .models import WordPair, Word

class WordPairSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = WordPair
        fields = ('pair1', 'pair2')

class WordSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Word
        fields = ('word')

