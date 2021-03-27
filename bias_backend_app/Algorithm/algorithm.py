import csv
import json
from nltk.corpus import stopwords
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import TweetTokenizer
from sklearn.decomposition import PCA


class biasAlgorithm:

    def __init__(self,model,type):
        self.__model = model
        self.__biased_word_pairs = self.__defaultBiasPair(type) #1->Gender 2->Age 3->Race
        self.__process_data()

    def estimate(self,index):
        if index == None:
            return "Unbiased"
        if(index >= 1.5 or index <= -1.5):
            return "High biased"
        elif(index >= 1.25 or index <= -1.25):
            return "Medium Biased"
        elif(index >= 1 or index <= -1):
            return "Low Biased"
        else:
            return "Unbiased"

    def __defaultBiasPair(self,index):
        if index == 1:#Gender
            return [('cowgirl', 'cowboy'), ('cowgirls', 'cowboys'), ('camerawoman', 'cameraman'), ('busgirl', 'busboy'), ('busgirls', 'busboys'), ('bellgirl', 'bellboy'), ('bellgirls', 'bellboys'), ('barwoman', 'barman'), ('barwomen', 'barmen'), ('seamstress', 'tailor'), ('seamstress', 'tailors'), ('princess', 'prince'), ('princesses', 'princes'), ('governess', 'governor'), ('governesses', 'governors'), ('adultress', 'adultor'), ('adultresses', 'adultors'), ('godess', 'god'), ('godesses', 'gods'), ('hostess', 'host'), ('hostesses', 'hosts'), ('abbess', 'abbot'), ('abbesses', 'abbots'), ('actress', 'actor'), ('actresses', 'actors'), ('spinster', 'bachelor'), ('spinsters', 'bachelors'), ('baroness', 'baron'), ('barnoesses', 'barons'), ('belle', 'beau'), ('belles', 'beaus'), ('bride', 'bridegroom'), ('brides', 'bridegrooms'), ('sister', 'brother'), ('sisters', 'brothers'), ('duchess', 'duke'), ('duchesses', 'dukes'), ('empress', 'emperor'), ('empresses', 'emperors'), ('enchantress', 'enchanter'), ('mother', 'father'), ('mothers', 'fathers'), ('fiancee', 'fiance'), ('fiancees', 'fiances'), ('nun', 'priest'), ('nuns', 'priests'), ('lady', 'gentleman'), ('ladies', 'gentlemen'), ('grandmother', 'grandfather'), ('grandmothers', 'grandfathers'), ('headmistress', 'headmaster'), ('headmistresses', 'headmasters'), ('heroine', 'hero'), ('heroines', 'heros'), ('lass', 'lad'), ('lasses', 'lads'), ('landlady', 'landlord'), ('landladies', 'landlords'), ('female', 'male'), ('females', 'males'), ('woman', 'man'), ('women', 'men'), ('maidservant', 'manservant'), ('maidservants', 'manservants'), ('marchioness', 'marquis'), ('masseuse', 'masseur'), ('masseuses', 'masseurs'), ('mistress', 'master'), ('mistresses', 'masters'), ('nun', 'monk'), ('nuns', 'monks'), ('niece', 'nephew'), ('nieces', 'nephews'), ('priestess', 'priest'), ('priestesses', 'priests'), ('sorceress', 'sorcerer'), ('sorceresses', 'sorcerers'), ('stepmother', 'stepfather'), ('stepmothers', 'stepfathers'), ('stepdaughter', 'stepson'), ('stepdaughters', 'stepsons'), ('stewardess', 'steward'), ('stewardesses', 'stewards'), ('aunt', 'uncle'), ('aunts', 'uncles'), ('waitress', 'waiter'), ('waitresses', 'waiters'), ('widow', 'widower'), ('widows', 'widowers'), ('witch', 'wizard'), ('witches', 'wizards'), ('airman', 'airwoman'), ('airmen', 'airwomen'), ('airwoman', 'airman'), ('airwomen', 'airmen'), ('girl', 'boy'), ('girls', 'boys'), ('brother', 'sister'), ('brothers', 'sisters'), ('businesswoman', 'businessman'), ('businesswomen', 'businessmen'), ('chairwoman', 'chairman'), ('chairwomen', 'chairman'), ('chick', 'dude'), ('chicks', 'dudes'), ('mom', 'dad'), ('moms', 'dads'), ('mommy', 'daddy'), ('mommies', 'daddies'), ('daughter', 'son'), ('daughters', 'sons'), ('gal', 'guy'), ('gals', 'guys'), ('lady', 'mentleman'), ('granddaughter', 'grandson'), ('granddaughters', 'grandsons'), ('groom', 'bride'), ('grooms', 'brides'), ('girl', 'guy'), ('girls', 'guys'), ('she', 'he'), ('himself', 'herself'), ('her', 'him'), ('herself', 'himself'), ('her', 'his'), ('wife', 'husband'), ('wives', 'husbands'), ('queen', 'king'), ('queens', 'kings'), ('lady', 'lord'), ('ladies', 'lords'), ("ma'am", 'sir'), ('miss', 'sir'), ('mrs.', 'mr.'), ('ms.', 'mr.'), ('policewoman', 'policeman'), ('prince', 'princess'), ('princes', 'princesses'), ('spokeswoman', 'spokesman'), ('spokeswomen', 'spokesmen'), ('womanly', 'manly'), ('girlish', 'boyish'), ('feminine', 'masculine'), ('Mary', 'John')]
        elif index == 2:#Age
            return [('ruth', 'taylor'), ('william', 'jamie'), ('horace', 'daniel'), ('mary', 'aubrey'), ('susie', 'alison'), ('amy', 'miranda'), ('john', 'jacob'), ('henry', 'arthur'), ('edward', 'aaron'), ('elizabeth', 'ethan'), ('wrinkled', 'unwrinkled'), ('wrinkled', 'placid'), ('unattractive', 'charming'), ('untidy', 'neat'), ('unattractive', 'attractive'), ('epicophosis', 'hearing'), ('deaf', 'hearing'), ('earless', 'hearing'), ('sedentary', 'energetic'), ('inactive', 'active'), ('sick', 'healthy'), ('fragile', 'strong'), ('tired', 'energized'), ('unhealthy', 'healthy'), ('inflexibility', 'flexibility'), ('constraint', 'flexibility'), ('old-fashioned', 'modern'), ('ancient', 'current'), ('olden', 'young'), ('old', 'young'), ('outdated', 'new'), ('outdated', 'fresh'), ('cautiousness', 'carelessness'), ('cautiousness', 'indiscretion'), ('self-discipline', 'blandness'), ('dejected', 'cheerful')]
        elif index == 3:#Racial
            return [('brad', 'darnell'), ('brendan', 'hakim'), ('geoffrey', 'jermaine'), ('greg', 'kareem'), ('brett', 'jamal'), ('matthew', 'leroy'), ('neil', 'tyrone'), ('todd', 'rasheed'), ('nancy', 'yvette'), ('amanda', 'malika'), ('emily', 'latonya'), ('rachel', 'jasmine'), ('alejandro', 'jermaine'), ('pancho', 'kareem'), ('bernardo', 'latonya'), ('pedro', 'jasmine'), ('octavio', 'hakim'), ('rodrigo', 'darnell'), ('ricardo', 'jamal'), ('augusto', 'leroy'), ('carmen', 'tyrone'), ('katia', 'rasheed'), ('marcella', 'yvette'), ('sofia', 'malika')]
        else: #Default is Gender
            return [('cowgirl', 'cowboy'), ('cowgirls', 'cowboys'), ('camerawoman', 'cameraman'), ('busgirl', 'busboy'), ('busgirls', 'busboys'), ('bellgirl', 'bellboy'), ('bellgirls', 'bellboys'), ('barwoman', 'barman'), ('barwomen', 'barmen'), ('seamstress', 'tailor'), ('seamstress', 'tailors'), ('princess', 'prince'), ('princesses', 'princes'), ('governess', 'governor'), ('governesses', 'governors'), ('adultress', 'adultor'), ('adultresses', 'adultors'), ('godess', 'god'), ('godesses', 'gods'), ('hostess', 'host'), ('hostesses', 'hosts'), ('abbess', 'abbot'), ('abbesses', 'abbots'), ('actress', 'actor'), ('actresses', 'actors'), ('spinster', 'bachelor'), ('spinsters', 'bachelors'), ('baroness', 'baron'), ('barnoesses', 'barons'), ('belle', 'beau'), ('belles', 'beaus'), ('bride', 'bridegroom'), ('brides', 'bridegrooms'), ('sister', 'brother'), ('sisters', 'brothers'), ('duchess', 'duke'), ('duchesses', 'dukes'), ('empress', 'emperor'), ('empresses', 'emperors'), ('enchantress', 'enchanter'), ('mother', 'father'), ('mothers', 'fathers'), ('fiancee', 'fiance'), ('fiancees', 'fiances'), ('nun', 'priest'), ('nuns', 'priests'), ('lady', 'gentleman'), ('ladies', 'gentlemen'), ('grandmother', 'grandfather'), ('grandmothers', 'grandfathers'), ('headmistress', 'headmaster'), ('headmistresses', 'headmasters'), ('heroine', 'hero'), ('heroines', 'heros'), ('lass', 'lad'), ('lasses', 'lads'), ('landlady', 'landlord'), ('landladies', 'landlords'), ('female', 'male'), ('females', 'males'), ('woman', 'man'), ('women', 'men'), ('maidservant', 'manservant'), ('maidservants', 'manservants'), ('marchioness', 'marquis'), ('masseuse', 'masseur'), ('masseuses', 'masseurs'), ('mistress', 'master'), ('mistresses', 'masters'), ('nun', 'monk'), ('nuns', 'monks'), ('niece', 'nephew'), ('nieces', 'nephews'), ('priestess', 'priest'), ('priestesses', 'priests'), ('sorceress', 'sorcerer'), ('sorceresses', 'sorcerers'), ('stepmother', 'stepfather'), ('stepmothers', 'stepfathers'), ('stepdaughter', 'stepson'), ('stepdaughters', 'stepsons'), ('stewardess', 'steward'), ('stewardesses', 'stewards'), ('aunt', 'uncle'), ('aunts', 'uncles'), ('waitress', 'waiter'), ('waitresses', 'waiters'), ('widow', 'widower'), ('widows', 'widowers'), ('witch', 'wizard'), ('witches', 'wizards'), ('airman', 'airwoman'), ('airmen', 'airwomen'), ('airwoman', 'airman'), ('airwomen', 'airmen'), ('girl', 'boy'), ('girls', 'boys'), ('brother', 'sister'), ('brothers', 'sisters'), ('businesswoman', 'businessman'), ('businesswomen', 'businessmen'), ('chairwoman', 'chairman'), ('chairwomen', 'chairman'), ('chick', 'dude'), ('chicks', 'dudes'), ('mom', 'dad'), ('moms', 'dads'), ('mommy', 'daddy'), ('mommies', 'daddies'), ('daughter', 'son'), ('daughters', 'sons'), ('gal', 'guy'), ('gals', 'guys'), ('lady', 'mentleman'), ('granddaughter', 'grandson'), ('granddaughters', 'grandsons'), ('groom', 'bride'), ('grooms', 'brides'), ('girl', 'guy'), ('girls', 'guys'), ('she', 'he'), ('himself', 'herself'), ('her', 'him'), ('herself', 'himself'), ('her', 'his'), ('wife', 'husband'), ('wives', 'husbands'), ('queen', 'king'), ('queens', 'kings'), ('lady', 'lord'), ('ladies', 'lords'), ("ma'am", 'sir'), ('miss', 'sir'), ('mrs.', 'mr.'), ('ms.', 'mr.'), ('policewoman', 'policeman'), ('prince', 'princess'), ('princes', 'princesses'), ('spokeswoman', 'spokesman'), ('spokeswomen', 'spokesmen'), ('womanly', 'manly'), ('girlish', 'boyish'), ('feminine', 'masculine'), ('Mary', 'John')]

    def __readBiasPair(self,address):
        pairlist = list(csv.reader(open(address,'r')))
        temp = []
        for x in pairlist:
           temp.append((x[0],x[1]))
        return temp

    def getBiasPair(self):
        return self.__biased_word_pairs

    def changeBiasPair(self,type):
        self.__biased_word_pairs =self.__defaultBiasPair(type)

    def getModel(self):
        return self.__model

    def changeModel(self,model):
        self.__model = model

    def __process_data(self):
        biasedPairs = []
        for pair in self.__biased_word_pairs:
            try:
                biasedPairs.append((self.__model[pair[0]], self.__model[pair[1]]))
            except:
                print(pair[0] + " and " + pair[1] + "are not in the word embedding")

        biases = []
        reversed_biases = []
        for pair in biasedPairs:
            biases.append(pair[0] - pair[1])
            reversed_biases.append(pair[1] - pair[0])
        self.__pca = PCA(n_components=1)
        processedBiasMatrix = biases + reversed_biases
        self.__pca.fit(np.array(processedBiasMatrix))
        tempOne = []
        tempTwo = []
        for pair in biasedPairs:
            tempOne.append(pair[0])
            tempTwo.append(pair[1])
        self.__bias_mean_one = np.mean(self.__pca.transform(np.array(tempOne)))
        self.__bias_mean_two = np.mean(self.__pca.transform(np.array(tempTwo)))

    def add_pair(self,newpair):
        self.__biased_word_pairs.extend(newpair)


    def __detect_bias_pca(self,word):
        if word not in self.__model:
            return None
        word_val = self.__pca.transform(np.array([self.__model[word]]))[0][0]
        return (word_val - (self.__bias_mean_two + self.__bias_mean_one) / 2) * 2 / (self.__bias_mean_two - self.__bias_mean_one)

    def detect(self,sentence):

        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(sentence)
        formattedTokens = [word.lower() for word in tokens]
        results = []
        stopwords = self.getStopWords()
        for token in formattedTokens:
                if token in stopwords:
                    index = 0
                    status = "Unbiased"
                else:
                    index = self.__detect_bias_pca(token)
                    status = self.estimate( index)
                if(status != "Unbiased" and status != "Low Biased"):
                    synonyms = self.getSynonyms(token)
                    for x in synonyms:
                        check = self.estimate(self.__detect_bias_pca(x))
                        if ( check == "High Biased"):
                            synonyms.remove(x)
                else:
                    synonyms = []
                token_result = {"original": tokens[formattedTokens.index(token)],"token": token, "bias":index,"status": status,"synonyms": synonyms,"index": formattedTokens.index(token) }                
                results.append(token_result)
        return json.dumps({"results": results},ensure_ascii=False)


    def getStopWords(self):
        stopWordsList = list(stopwords.words('english'))
        sentimentWord = ['he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself']
        for y in sentimentWord:
            stopWordsList.remove(y)
        return stopWordsList

    def getSynonyms(self,word):
        synonyms = set()
        #wordnet
        #nltk.download('wordnet')
        synList = wordnet.synsets(word)
        for x in wordnet.synsets(word):
            for y in x.lemmas():
                synonym = y.name()#.replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

if __name__ == "__main__":
    vv = biasAlgorithm(None)
    print(vv.getBiasPair())
