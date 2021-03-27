from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import render

from rest_framework import viewsets
from .serializers import WordPairSerializer, WordSerializer
from .models import WordPair, Word, UserConfig
from rest_framework.views import APIView
import json
from .Algorithm import controller
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import pytesseract
try:
    from PIL import Image
except:
    import Image

import logging
import base64
import io

from pdf2image import convert_from_bytes
# Create your views here.

class WordPairViewSet(viewsets.ModelViewSet):
    queryset = WordPair.objects.all().order_by('pair1')
    serializer_class  = WordPairSerializer

class WordViewSet(viewsets.ModelViewSet):
    queryset = Word.objects.only('word')
    print(queryset)
    serializer_class  = WordSerializer



class handleRequest(APIView):
    @csrf_exempt
    def post(self, request):
        logger = logging.getLogger('test')

        requestBody = None
        if "application/json" in request.META['CONTENT_TYPE']:
            requestBody = json.loads(request.body)
        else:
            requestBody = request.POST.get('requestBody')
        # print(requestBody.keys())

        file_string = requestBody['file_obj']
        preference = requestBody['preference']
        model = requestBody['model']
        #model_url = requestBody['model_url']

        decoded_str = ""

        # print(file_string)
        if "data:text/plain" in file_string:
            decoded_str = self._process_text(file_string)
        elif "data:application/pdf" in file_string:
            decoded_str = self._process_pdf(file_string)
        elif "data:image/jpeg" in file_string:
            decoded_str = self._process_img(file_string)
        else:
            return JsonResponse({
                'status': 'false',
                'message': "ERROR: unsupported file type"
            }, status=400)


        cc = controller.maincontroller() #create instance
        #if model1 or model2 or model3:
        #   cc.setType(0)
        #elif corpus1 or corpus2 or corpus3:
        #   cc.setType(1)
        
        #cc.setType(2) #use url local model

        if preference == 'gender':
            cc.setBiasPair(1)
        elif preference == 'race':
            cc.setBiasPair(2)
        elif preference == 'age':
            cc.setBiasPair(3)
        else:
            cc.setBiasPair(1) #default is Gender type
        
        #defalut model is GoogleNews
        #Change the choosen_model_address path to GoogleNewsModel location
        choosen_model_address = "/Users/markhan/UCL_CS/System_Engineering/final/bias-detect/bias_backend/bias_backend/bias_backend/bias_backend_app/Algorithm/GoogleNews-vectors-negative300.bin.gz"

        if model == 'model1': #model1 -> GoogleNews
            # cc.setType(2)
            # choosen_model_address = "/Users/markhan/UCL_CS/System_Engineering/final/bias-detect/bias_backend/bias_backend/bias_backend/bias_backend_app/Algorithm/GoogleNews-vectors-negative300.bin.gz"
            # cc.changeUrl(choosen_model_address)
            cc.setType(0)
            cc.setModelSelection(3)

        elif model == 'model2': 
            cc.setType(0)
            cc.setModelSelection(0)

        elif model == 'model3': 
            cc.setType(0)
            cc.setModelSelection(1)

        elif model == 'model4':
            cc.setType(0)
            cc.setModelSelection(2)

        else: #When user send URL
            cc.setType(2)
            cc.changeUrl(model)

        cc.initialise() #init model and algo

        bias_str = cc.processSentence(decoded_str)

        print(bias_str)
        
        return JsonResponse({
            'status': "OK",
            'bias_str': bias_str,
            'decoded_str': decoded_str
        })

    def _process_text(self, file_string):
        file_string = file_string.split('base64,')[-1].strip()
        pic = io.StringIO()

        byte_base64 = base64.b64decode(file_string)
        print(byte_base64)
        decoded = byte_base64.decode('utf-8')
        
        print(decoded)

        return decoded


    def _process_pdf(self, file_string):
        file_string = file_string.split('base64,')[-1].strip()


        pic = io.StringIO()
        images = convert_from_bytes(base64.b64decode(file_string))

        ocr_str = ""
        for image in images:

            # bg = Image.new("RGB", image.size, (255,255,255))
            # bg.paste(image,image)

            ocr_str += pytesseract.image_to_string(image)
        
        return ocr_str
    
    def _process_img(self, file_string):
        file_string = file_string.split('base64,')[-1].strip()


        pic = io.StringIO()
        image_string = io.BytesIO(base64.b64decode(file_string))

        image = Image.open(image_string)
        bg = Image.new("RGB", image.size, (255,255,255))
        bg.paste(image,image)
        
        ocr_str = pytesseract.image_to_string(bg)

        return ocr_str
