#LOO_ML.py       #2021.04.21
#hur.benjamin@mayo.edu
#
#Do simple Machine learning based on LOO-CV
#logistic regression default cutoff (>=0.5)
#Random Forest
#SVM

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def count_for_confusion_matrix(y_test, y_pred, count_1, count_2, count_3, count_4):
    y_test = int(y_test)
    y_pred = int(y_pred)
    
    if y_test == 1:    #obs MCII+
        if y_pred == 1:  #pred MCII+
            count_3 += 1
        else:            #pred MCII-
            count_4 += 1
        
    if y_test == 0:     #obs MCII-
        if y_pred == 1:   #pred MCII+
            count_1 += 1
        else:             #pred MCII-
            count_2 += 1
    return count_1, count_2, count_3, count_4


def make_confusion_matrix(count_1, count_2, count_3, count_4):
#Predicted MCII, MCII-
#observed MCII-  count 1, count 2
#observed MCII+  count 3, count 4

    tp =  count_2 / (count_2 + count_1)
    tn =  count_3 / (count_3 + count_4)
    acc = count_3 + count_2 / (count_1 + count_2 + count_3 + count_4)
    bal_acc = (tp + tn) / 2

    print ('       #PRED_MCII+   #PRED_MCII-')
    print ('obs_MCII-    %s            %s' % (count_1, count_2))
    print ('obs_MCII+    %s            %s' % (count_3, count_4))
    print ('TP: %s' % tp)
    print ('TN: %s' % tn)
    print ('ACC: %s' % acc)
    print ('bal ACC: %s' % bal_acc)

def main(data_df, specified_classifier):
    r, c = data_df.shape
    diff_count = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0

    for i in range(r):
#         print ('%s/%s' % (i,r))
        y_test = data_df.iloc[i,0]
        X_test = data_df.iloc[i,1:]

        train_df = data_df.copy()
        train_df.drop(index=train_df.iloc[i].name, inplace=True)

        y_train = train_df.iloc[:,0]
        X_train = train_df.iloc[:,1:]

        clf = specified_classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict([X_test])

        if int(y_test) != int(y_pred):
            diff_count += 1
#         print ("Answer: %s, Predicted: %s" % (y_test, y_pred))

        count_1, count_2, count_3, count_4 = count_for_confusion_matrix(y_test, y_pred, 
                                                                        count_1, count_2, count_3, count_4)

    print ('False Predicted: %s' % diff_count)
    return count_1, count_2, count_3, count_4


if __name__  == "__main__":	

	data_file = '../data/RA_microbiome_data.ml_ready.matrix.tsv'
	data_df = pd.read_csv(data_file, sep='\t', index_col=0)
	data_df = data_df.T


	#LR
	print ('LOO-CV, logistic regression')
	specified_classifier = LogisticRegression(max_iter=1000) #default is 100
	count_1, count_2, count_3, count_4 = main(data_df, specified_classifier)
	make_confusion_matrix(count_1, count_2, count_3, count_4)

	#RF
	print ('LOO-CV, random forest')
	specified_classifier = RandomForestClassifier()
	count_1, count_2, count_3, count_4 = main(data_df, specified_classifier)
	make_confusion_matrix(count_1, count_2, count_3, count_4)

	#SVM
	print ('LOO-CV, SVM')
	specified_classifier = svm.SVC()
	count_1, count_2, count_3, count_4 = main(data_df, specified_classifier)
	make_confusion_matrix(count_1, count_2, count_3, count_4)

else:
	print ("Not meant to be called")
