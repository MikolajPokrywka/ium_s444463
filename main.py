import pandas as pd
import numpy as np
import scipy
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import kaggle
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch import optim
import matplotlib.pyplot as plt


def convert_text_to_model_form(text):
    a = vectorizer.transform([text])
    b = torch.tensor(scipy.sparse.csr_matrix.todense(a)).float()
    return b


if __name__ == "__main__":
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('shivamb/real-or-fake-fake-jobposting-prediction', path='.',
                                      unzip=True)

    data = pd.read_csv('fake_job_postings.csv', engine='python')
    # data = data.replace(np.nan, '', regex=True)
    data = data[["company_profile", "fraudulent"]]
    data = data.dropna()

    data_train, data_test = train_test_split(data, test_size=3000, random_state=1)
    data_dev, data_test = train_test_split(data_test, test_size=1500, random_state=1)

    x_train = data_train["company_profile"]
    x_dev = data_dev["company_profile"]
    x_test = data_test["company_profile"]

    y_train = data_train["fraudulent"]
    y_dev = data_dev["fraudulent"]
    y_test = data_test["fraudulent"]

    x_train = np.array(x_train)
    x_dev = np.array(x_dev)
    x_test = np.array(x_test)

    y_train = np.array(y_train)
    y_dev = np.array(y_dev)
    y_test = np.array(y_test)

    vectorizer = TfidfVectorizer()

    x_train = vectorizer.fit_transform(x_train)
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
        nn.Linear(64, data_train["fraudulent"].nunique()),
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

    epochs = 50
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
    log_ps = model(x_test)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    descr = np.array(data_test["company_profile"])
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

    plt.figure(figsize=(12, 5))
    ax = plt.subplot(121)
    plt.xlabel('epochs')
    plt.ylabel('negative log likelihood loss')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.subplot(122)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.plot(test_accuracies)
    plt.show()
