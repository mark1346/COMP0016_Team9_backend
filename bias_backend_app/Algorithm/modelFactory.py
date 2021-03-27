import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec


class model:
    def __init__(self):

        info = api.info()
        self.__type = 0 #default
        self.__ListSelection = 5 #default
        self.__model = None
        self.__address = "/Users/markhan/UCL_CS/System_Engineering/final/bias-detect/bias_backend/bias_backend/bias_backend/bias_backend_app/Algorithm/GoogleNews-vectors-negative300.bin.gz" #default
        self.__preTrainedModelList = list(info['models'].keys())
        self.__corporaList = list(info['corpora'].keys())

    def getPretrainedModelList(self):
        return self.__preTrainedModelList

    def getCorporaList(self):
        return self.__corporaList

    def getModel(self):
        return self.__model

    def getType(self):
        return self.__type

    def getSelect(self):
        return self.__ListSelection

    def getlocalModelAddress(self):
        return self.__address
        
#type = 0 pretrained model, type = 1 online training model
    def setType(self,type):
        self.__type = type

    def setSelect(self,select):
        self.__ListSelection = select

    def setlocalModelAddress(self,address):
        self.__address = address

    # def trainModel(self):
    #     corpus = api.load(self.__corporaList[6])
    #     self.__model = Word2Vec(corpus).wv


    def generateModel(self):
        if self.__type == 0:
            self.__model = api.load(self.__preTrainedModelList[int(self.__ListSelection)])
        elif self.__type == 1:
            corpus = api.load(self.__corporaList[int(self.__ListSelection)])
            self.__model = Word2Vec(corpus).wv
        elif self.__type == 2:
            self.__model = KeyedVectors.load_word2vec_format(self.__address, binary=True)


