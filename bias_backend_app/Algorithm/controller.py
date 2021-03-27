from .algorithm import biasAlgorithm
from .modelFactory import model
from .ocr import OCR


class maincontroller:
    def __init__(self):
        self.__modelFactory = model()
        self.__algo = None
        self.__biasPairType = None
        self.__result = {"results": [{"original": "ui", "token": "cv", "bias": 0.7547706271041127, "status": "Unbiased", "synonyms": []}]}
        #Testing default result

# Model related functions
    def setType(self,type): #Set model type : type = 0 online pretrained model, type = 1 online training corpus, type = 2 Url pretrained model
        self.__modelFactory.setType(type)

    def setModelSelection(self,select): #set model of corpus selection based on the lists returned by method: getPretrainedModelList() and getCorporaList()
        self.__modelFactory.setSelect(select)

    def getModel(self):# return model in factory
        return self.__modelFactory.getModel()

    def getPretrainedModelList(self):# get name list of pretrained model
        return self.__modelFactory.getPretrainedModelList()

    def getCorporaList(self):# get name list of corpus that used to train model
        return self.__modelFactory.getCorporaList()

    def changeModel(self):
        self.modelSetting()
        self.__algo.changeModel(self,self.__modelFactory.getModel)

    def changeUrl(self,address):
        self.__modelFactory.setlocalModelAddress(address)

    def modelSetting(self): #initialised the model & change the model
        #type = 0 online pretrained model, type = 1 online training corpus, type = 2 Url pretrained model
        self.__modelFactory.generateModel()

#----------------------------------------

#Algorithm related functions
    def algorithm_init(self): #initialise the algorithm structure
        self.__algo = biasAlgorithm(self.__modelFactory.getModel(),self.__biasPairType)

    #Bias pair related functions
    def setBiasPair(self,type): #change bias pair by given it bias csv location
        self.__biasPairType =type

    def addBiasPair(self,biasPair):#add bias pair in current storage, but not change the csv file
        self.__algo.add_pair(biasPair)
    #------------------------------

#----------------------------------------

#OCR function
    def readimage(self,address):# read character from image
        self.ocr = ocr.OCR()
        return self.ocr.readimage(address)
#----------------------------------------

#Running functions
    def initialise(self):
        self.modelSetting()
        self.algorithm_init()

    def processSentence(self,sentence):
        result = self.__algo.detect(sentence)
        self.__result = result
        return result

    def run_example1(self):
        cc = maincontroller()#create instance
        cc.setType(2)#use url local model
        cc.setBiasPair(1)#set Gender bias pair
        cc.initialise()#init model and algo
        sentence = cc.readimage("cv_example1.pdf") #read input from OCR
        print(cc.processSentence(sentence))

    def run_example2(self):
        cc = maincontroller()#create instance
        cc.setType(2)#use url local model
        cc.setBiasPair(1)#set Gender bias pair
        cc.initialise()#init model and algo
        sentence = "girl is an actress" #read input from OCR
        print(cc.processSentence(sentence))
#----------------------------------------


#UI FUNCTION PART
    def getResult(self):
        return self.__result
    def displayContent(self):
        displayText = ""
        for tokens in self.__result["results"]:
            displayText += " " + tokens["original"]
        return displayText

    def changeContent(self,listSubstitute):
        #listSubstitute = [[0,3,5],["fsad","gdfd","fgfdf"]]
        for x in range(0,listSubstitute[0]):
            index = listSubstitute[0][x]
            content = listSubstitute[1][x]
            self.__result["results"][index]["original"] = content
#----------------------------------------


if __name__ == "__main__":
    rr = maincontroller()
    print(rr.run_example2())
