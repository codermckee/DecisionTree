# -*- coding: utf-8 -*-
'''
@author: mckee
'''
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
if __name__ == '__main__':
    attributeList = []
    labelList = []
    handle = csv.reader(open('data.csv'))
    n = 0
    for line in handle:
        if n == 0:
            attribute = line
        else:
            if line[-1] == 'yes':
                labelList.append(1)
            else:
                labelList.append(0)
            feature = {}
            feature[attribute[1]] = line[1]
            feature[attribute[2]] = line[2]
            feature[attribute[3]] = line[3]
            feature[attribute[4]] = line[4]
            attributeList.append(feature)
        n = n+1
    vec = DictVectorizer()
    feature_Matrix = vec.fit_transform(attributeList).toarray()
    describe = ['middle_aged','senior','youth','excellent','fair','high','low','medium','no','yes']
    print feature_Matrix
    print describe
    print labelList
    classifier = tree.DecisionTreeClassifier(criterion='entropy')
    print classifier #分类器的默认参数
    classifier = classifier.fit(feature_Matrix,labelList)

    handle1 = open('decisionTREE.txt','w')
    handle1 =  tree.export_graphviz(classifier,feature_names = vec.get_feature_names(),out_file=handle1)

    test = feature_Matrix[2]
    predict = classifier.predict(test)
    print predict


