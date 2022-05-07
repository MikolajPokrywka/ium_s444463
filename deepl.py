import pandas as pd
import numpy as np
import scipy
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
# import kaggle
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import sys
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()

ex.observers.append(FileStorageObserver('my_runs'))
vectorizer = TfidfVectorizer()

@ex.config
def my_config():
    epochs = 10


def convert_text_to_model_form(text):
    a = vectorizer.transform([text])
    b = torch.tensor(scipy.sparse.csr_matrix.todense(a)).float()
    return b

@ex.automain
def my_main(epochs, _run):
    # print(sys.argv[1])
    # print(type(sys.argv[1]))
    # print(sys.argv[1])
    # epochs = int(sys.argv[1])
    # epochs=10

    # kaggle.api.authenticate()
    # kaggle.api.dataset_download_files('shivamb/real-or-fake-fake-jobposting-prediction', path='.',
    #                                   unzip=True)

    data = pd.read_csv('fake_job_postings.csv', engine='python')
    # data = data.replace(np.nan, '', regex=True)
    data = data[["company_profile", "fraudulent"]]
    data = data.dropna()
    company_profile = data["company_profile"]

    # data_train, data_test = train_test_split(data, test_size=3000, random_state=1)
    # data_dev, data_test = train_test_split(data_test, test_size=1500, random_state=1)
    data_train = pd.read_csv('data_train.csv', engine='python', header=None).dropna()
    data_dev = pd.read_csv('data_dev.csv', engine='python', header=None).dropna()
    data_test = pd.read_csv('data_test.csv', engine='python', header=None).dropna()

    x_train = data_train[5]
    x_dev = data_dev[5]
    x_test = data_test[5]

    y_train = data_train[17]
    y_dev = data_dev[17]
    y_test = data_test[17]

    company_profile = np.array(company_profile)
    x_train = np.array(x_train)
    x_dev = np.array(x_dev)
    x_test = np.array(x_test)

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)


    company_profile = vectorizer.fit_transform(company_profile)
    x_train = vectorizer.transform(x_train)
    x_dev = vectorizer.transform(x_dev)
    x_test = vectorizer.transform(x_test)

    x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
    x_dev = torch.tensor(scipy.sparse.csr_matrix.todense(x_dev)).float()
    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

    y_train = torch.tensor(y_train)
    y_dev = torch.tensor(y_dev)
    y_test = torch.tensor(y_test)

    from torch import nn

    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, data_train[17].nunique()),
        nn.LogSoftmax(dim=1))

    # Define the loss
    criterion = nn.NLLLoss()  # Forward pass, log
    logps = model(x_train)  # Calculate the loss with the logits and the labels
    loss = criterion(logps, y_train)
    loss.backward()  # Optimizers need parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for e in range(epochs):
        optimizer.zero_grad()

        output = model.forward(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        train_loss = loss.item()
        train_losses.append(train_loss)

        optimizer.step()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            model.eval()
            log_ps = model(x_dev)
            test_loss = criterion(log_ps, y_dev)
            test_losses.append(test_loss)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == y_dev.view(*top_class.shape)
            test_accuracy = torch.mean(equals.float())
            test_accuracies.append(test_accuracy)

        model.train()

        print(f"Epoch: {e + 1}/{epochs}.. ",
              f"Training Loss: {train_loss:.3f}.. ",
              f"Test Loss: {test_loss:.3f}.. ",
              f"Test Accuracy: {test_accuracy:.3f}")

    TP = []
    TF = []

    FP = []
    FN = []
    model.eval()
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
    print(f"F- score = {f_score}")
    f = open("model_resutls.txt", "a")

    f.write(f"F-SCORE = {f_score}\n")
    f.write(f"TP = {len(TP)}\n")
    f.write(f"TF = {len(TF)}\n")
    f.write(f"FP = {len(FP)}\n")
    f.write(f"FN = {len(FN)}\n")

    f.write(f"TP descriptions:")
    for i in TP:
        f.write(i+'\n')
    f.write(f"TF descriptions:")
    for i in TF:
        f.write(i+"\n")
    f.write(f"FP descriptions:")
    for i in FP:
        f.write(i+"\n")
    f.write(f"FN descriptions:")
    for i in FN:
        f.write(i+"\n")
    f.close()
    
    torch.save(model, 'model')
    ex.add_artifact("model")


    # plt.figure(figsize=(12, 5))
    # ax = plt.subplot(121)
    # plt.xlabel('epochs')
    # plt.ylabel('negative log likelihood loss')
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(test_losses, label='Validation loss')
    # plt.legend(frameon=False)
    # plt.subplot(122)
    # plt.xlabel('epochs')
    # plt.ylabel('test accuracy')
    # plt.plot(test_accuracies)
    # plt.show()
