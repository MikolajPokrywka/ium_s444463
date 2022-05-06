import torch
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import matplotlib.pyplot as plt
import re

if __name__ == "__main__":
    data = pd.read_csv('fake_job_postings.csv', engine='python')
    # data = data.replace(np.nan, '', regex=True)
    company_profile = data["company_profile"]
    company_profile = company_profile.dropna()
    company_profile = np.array(company_profile)
    vectorizer = TfidfVectorizer()

    company_profile = vectorizer.fit_transform(company_profile)
    model = torch.load('model')

    data_test = pd.read_csv('data_test.csv', engine='python', header=None)
    data_test = data_test.dropna()
    x_test = data_test[5]
    y_test = data_test[17]


    x_test = np.array(x_test)

    y_test = np.array(y_test)


    x_test = vectorizer.transform(x_test)

    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

    y_test = torch.tensor(y_test)



    TP = []
    TF = []

    FP = []
    FN = []
    # x_test = x_test.view(x_test.size(0), -1)

    model = model.eval()
    print(x_test.size())
    log_ps = model(x_test)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    descr = np.array(data_test[5])
    for i, (x, y) in enumerate(zip(np.array(top_class), np.array(y_test.view(*top_class.shape)))):
        d = descr[i]
        if x == y:
            if x:
                TP.append(d)
            else:
                TF.append(d)
        else:
            if x:
                FP.append(d)
            else:
                FN.append(d)
    f_score = len(TP) / (len(TP) + 0.5 * (len(FP) + len(FN)))
    accuracy = (len(TP) + len(TF)) / (len(TP) + len(TF) + len(FP) + len(FN))  
    precision = len(TP) / ( len(TP) + len(FP) )
    recall = len(TP) / ( len(TP) + len(FN) )
    print(f"F- score = {f_score}")
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    f = open("metrics.txt", "a")

    f.write(f"F-SCORE = {f_score}\n")
    f.write(f"Accuracy = {accuracy}\n")
    f.write(f"Precision = {precision}\n")
    f.write(f"Recall = {recall}\n")

    f.close()
    f_read = open("metrics.txt", "r")
    content = re.findall('F-SCORE = [0-9.]+', f_read.read())
    fscores = []
    for c in content:
        r = re.findall("\d+\.\d+", c)
        fscores.append(float(r[0]))
    
    plt.ylabel('F score')
    plt.plot(np.arange(0, len(fscores)), fscores) 
    # plt.xticks(np.arange(0, len(fscores)+1, 5))
    plt.savefig('metrics.png')
    # f.write(f"TP descriptions:")
    # for i in TP:
    #     f.write(i+'\n')
    # f.write(f"TF descriptions:")
    # for i in TF:
    #     f.write(i+"\n")
    # f.write(f"FP descriptions:")
    # for i in FP:
    #     f.write(i+"\n")
    # f.write(f"FN descriptions:")
    # for i in FN:
    #     f.write(i+"\n")
    # f.close()
    a=1