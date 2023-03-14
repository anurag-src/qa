import sys
import os
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import numpy as np
import math
from nltk.stem import WordNetLemmatizer
from scipy import spatial
import gensim
from gensim.models import Word2Vec
import spacy
from spacy import displacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

#nlp = en_core_web_sm.load()

lemmatizer = WordNetLemmatizer()
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = stopwords.words('english')
porter = PorterStemmer()


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


def preprocess(s):
    s = s.lower()
    s_new = ''
    for i in range(len(s)):
        if s[i] not in string.punctuation:
            s_new = s_new+ s[i]
        if s[i] == '.':
            s_new = s_new + s[i]
    return s_new


def index(word, arr):
    for i in range(len(arr)):
        if word == arr[i]:
            return i
    return -1


def prequestion(question):
    question = question.lower()
    question_new = ''
    for i in range(len(question)):
        if question[i] not in string.punctuation:
            question_new = question_new + question[i]
    question_new = word_tokenize(question_new)
    question_new = [word for word in question_new if word not in stop_words]
    question_new = [lemmatizer.lemmatize(word) for word in question_new]
    return question_new


def vectorize(s, vocab):
    vec = np.zeros(len(vocab), dtype = int)
    for i in range(len(s)):
        if s[i] in vocab:
            vec[index(s[i],vocab)] = vec[index(s[i],vocab)] + 1
    return vec


# def sentence_embedding(sentence,model):
#     embedding = np.zeros(200, dtype = float)
#     count = 0
#     for word in range(len(sentence)):
#         if (sentence[word] in model.wv.key_to_index) and (sentence[word] not in stop_words):
#             add = np.copy(model.wv.get_vector(sentence[word]))
#             embedding = embedding + add
#             count += 1
#     if count == 0:
#         return embedding
#     return embedding/count
#
# def question_embedding(question,model):
#     question = question.lower()
#     question_new = ''
#     count = 0
#     for i in range(len(question)):
#         if question[i] not in string.punctuation:
#             question_new = question_new + question[i]
#     question_new = word_tokenize(question_new)
#     embedding = np.zeros(200, dtype = float)
#     for word in range(len(question_new)):
#         if (question_new[word] in model.wv.key_to_index) and (question_new[word] not in stop_words):
#             add = np.copy(model.wv.get_vector(question_new[word]))
#             embedding = embedding + add
#             count += 1
#     if count == 0:
#         return embedding
#     return embedding/count
#
#
# emmbed_dict = {}
# with open('glove.6B.100d.txt','r') as f:
#   for line in f:
#     values = line.split()
#     word = values[0]
#     vector = np.asarray(values[1:],'float32')
#     emmbed_dict[word]=vector
#
# def glove_embedding(sentence):
#     embedding = np.zeros(100, dtype=float)
#     count = 0
#     for word in range(len(sentence)):
#         if word in emmbed_dict:
#             embedding = embedding + emmbed_dict[sentence[word]]
#             count += 1
#         else:
#             if (sentence[word] in model1.wv.key_to_index) and (sentence[word] not in stop_words):
#                 add = np.copy(model1.wv.get_vector(sentence[word]))
#                 embedding = embedding + add
#                 count += 1
#     return np.array(embedding/count)
#
# def glove_embeddingq(question):
#     question = question.lower()
#     question_new = ''
#     count = 0
#     for i in range(len(question)):
#         if question[i] not in string.punctuation:
#             question_new = question_new + question[i]
#     question_new = word_tokenize(question_new)
#     embedding = np.zeros(100, dtype=float)
#     for word in range(len(question_new)):
#         if word in emmbed_dict:
#             embedding = embedding + emmbed_dict[question_new[word]]
#             count += 1
#         else:
#             if (question_new[word] in model1.wv.key_to_index) and (question_new[word] not in stop_words):
#                 add = np.copy(model1.wv.get_vector(question_new[word]))
#                 embedding = embedding + add
#                 count += 1
#     if count == 0:
#         return embedding
#     return np.array(embedding/count)

op = open('ogresponsefull1','w')
file = open(sys.argv[1])
file = file.readlines()
path = file[0].replace('\n', '')
stories = []
st = ".story"
ans = ".answers"
qs = ".questions"


# train = open('training.txt').read()
# train = sent_tokenize(train)
# data = []
# for i in range(len(train)):
#     train[i] = train[i].replace('.','')
#     train[i] = word_tokenize(train[i])
#     train[i] = [word for word in train[i] if word not in stop_words]
#     data.append(train[i])
# model1 = gensim.models.Word2Vec(sentences= data, min_count = 1, vector_size = 100, window = 5)
# model2 = gensim.models.Word2Vec(sentences= data, min_count = 1, vector_size = 100,
#                                              window = 5, sg = 1)

for i in range(1, len(file)):
    if file[i] != '\n':
        stories.append(file[i].replace('\n', ''))

# for i in range(74,113):
#     questions = []
#     sentences = []
#     s = path + "/" + stories[i]
#     q = s + qs
#     s = s + st
#     file = open(q).readlines()
#     for j in range(len(file)):
#         file[j] = file[j].split(": ")
#         if file[j][0] == "Question":
#             questions.append([file[j-1][0] + ": " + file[j-1][1].replace("\n",""),file[j][1].replace("\n","")])
#     story = str(open(s).read())
#     story = story.split('TEXT:\n\n')
#     story = story[1]
#     story = preprocess(story)
#     story = story.replace('\n', ' ')
#     story = sent_tokenize(story)
#     for j in range(len(story)):
#         sentences.append(story[j])
#         story[j] = story[j].replace('.','')
#         story[j] = word_tokenize(story[j])
#         #story[j] = sentence_embedding(story[j],model1)
#         story[j] = glove_embedding(story[j])
#     for j in range(len(questions)):
#         answer = 'none'
#         max = 0
#         #questions[j][1] = question_embedding(questions[j][1],model1)
#         questions[j][1] = glove_embeddingq(questions[j][1])
#         if magnitude(questions[j][1]) == 0:
#             answer = ''
#         for k in range(len(sentences)):
#             if magnitude(questions[j][1]) != 0 and magnitude(story[k])!=0:
#                 val = 1- spatial.distance.cosine(questions[j][1],story[k])
#                 if val > max:
#                     max = val
#                     answer = sentences[k]
#
#         op.write(questions[j][0] + '\n')
#         if answer != '':
#             temp = word_tokenize(answer)
#             op.write("Answer: ")
#             for p in range(1, len(temp) - 1):
#                 op.write(temp[p] + " ")
#             op.write('\n\n')
#         else:
#             op.write("Answer: " + answer + '\n\n')


def question_type(question):
    question = question.lower()
    question_new = ''
    for i in range(len(question)):
        if question[i] not in string.punctuation:
            question_new = question_new + question[i]
    question_new = word_tokenize(question_new)
    if question_new[0] == "where":
        return 'where'
    if question_new[0] == "who":
        return 'who'
    if question_new[0] == "when":
        return 'when'
    else:
        return "none"

WHO = ["PERSON","ORG","NORP"]
WHERE = ["GPE","LOC"]
WHEN = ["DATE","TIME"]

def ner(sentence,type):
    arr = []
    answer = ''
    if type == 'none':
        return False, answer
    if type == "where":
        arr = WHERE
    if type == "who":
        arr = WHO
    if type == "when":
        arr = WHEN
    count = 0
    doc = nlp(sentence)
    for x in doc:
        if x.ent_type_ in arr:
            count += 1
            answer = answer + x.text
    if count != 0:
        return True, answer
    else:
        return False, answer

for i in range(len(stories)):
    questions = []
    sentences = []
    s = path + "/" + stories[i]
    q = s + qs
    s = s + st
    file = open(q).readlines()
    for j in range(len(file)):
        file[j] = file[j].split(": ")
        if file[j][0] == "Question":
            questions.append([file[j-1][0] + ": " + file[j-1][1].replace("\n",""),file[j][1].replace("\n","")])
    story = str(open(s).read())
    story = story.split('TEXT:\n\n')
    story = story[1]
    story = preprocess(story)
    story = story.replace('\n', ' ')
    story = sent_tokenize(story)
    for j in range(len(story)):
        sentences.append(story[j])
        story[j] = story[j].replace('.','')
        story[j] = word_tokenize(story[j])
        story[j] = [word for word in story[j] if word not in stop_words]
        story[j] = [lemmatizer.lemmatize(word) for word in story[j]]
    # if i == 0:
    #     print(sentences[0])
    #     doc = nlp(sentences[0])
    #     for ent in doc:
    #         print(ent.text, ent.ent_iob_, ent.ent_type_)
    for j in range(len(questions)):
        max = 0
        answer = 'none'
        type = question_type(questions[j][1])
        #print(type)
        vocab = prequestion(questions[j][1])
        question_vec = vectorize(vocab, vocab)
        #if i == 3 and j == 1:
            #print(vocab)
        for k in range(len(sentences)):
            answer_vec = vectorize(story[k],vocab)
            if magnitude(question_vec) != 0 and magnitude(answer_vec)!=0:
                val = 1- spatial.distance.cosine(answer_vec,question_vec)
                if val > max:
                    max = val
                    answer = sentences[k]

        #print(questions[j][0])
        #op.write("Answer: " + answer + '\n\n')
        op.write(questions[j][0] + '\n')
        # bool, neranswer = ner(answer, type)
        # if bool:
        #     op.write("Answer: " + neranswer + '\n\n')
        if answer != '':
            temp = word_tokenize(answer)
            s = "Answer: "
            for p in range(1,len(temp)-1):
                s = s + temp[p] + " "
            #print(s + '\n')
            op.write(s + '\n\n')
        else:
            print("Answer: " + answer + '\n')