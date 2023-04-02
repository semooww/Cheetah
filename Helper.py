import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
import os


# Display the distribution of data on the train and test
def plot_pie_sets(arrays, technique, model_info):
    MAIN_PATH = r"Plots"
    os.makedirs(MAIN_PATH, exist_ok=True)
    MODEL_PATH = os.path.join(MAIN_PATH, model_info)
    os.makedirs(MODEL_PATH, exist_ok=True)
    PATH = os.path.join(MODEL_PATH, f"{technique}_pie.png")

    titles = ["Train Set", "Test Set"]
    labels = ["Normal", "Cataract", "Glaucoma"]
    fig = plt.figure(figsize=(12, 5))
    plt.title(f"Distribution of {technique}")
    plt.axis('off')
    plt.grid(False)
    for i in range(2):
        fig.add_subplot(1, 2, i + 1)
        arr = []
        for j in range(3):
            arr.append((arrays[i] == j).sum())
        plt.title(titles[i])
        plt.pie(arr, autopct=lambda x: '{:.0f}'.format(x * np.array(arr).sum() / 100))
        plt.legend(labels=labels, loc=0)
    plt.savefig(PATH)


def plot_history(arrays, technique, model_info):
    MAIN_PATH = r"Plots"
    os.makedirs(MAIN_PATH, exist_ok=True)
    MODEL_PATH = os.path.join(MAIN_PATH, model_info)
    os.makedirs(MODEL_PATH, exist_ok=True)
    PATH = os.path.join(MODEL_PATH, f"{technique}_history.png")

    fig = plt.figure(figsize=(15, 6))
    plt.title(f"Evaluation for {technique} Images")
    plt.axis('off')
    plt.grid(False)

    fig.add_subplot(1, 2, 1)
    plt.plot(arrays[0])
    plt.plot(arrays[1])

    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend(["Train", "Validation"], loc=0)

    fig.add_subplot(1, 2, 2)
    plt.plot(arrays[2])
    plt.plot(arrays[3])

    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend(["Train", "Validation"], loc=0)
    plt.savefig(PATH)


def plot_conf_matrix(model, X_test, y_test, technique, model_info):
    MAIN_PATH = r"Plots"
    os.makedirs(MAIN_PATH, exist_ok=True)
    MODEL_PATH = os.path.join(MAIN_PATH, model_info)
    os.makedirs(MODEL_PATH, exist_ok=True)
    PATH = os.path.join(MODEL_PATH, f"{technique}_confusion_matrix.png")

    labels = ["Normal", "Cataract", "Glaucoma"]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.title(f"Confusion Matrix of {technique}")
    plt.savefig(PATH)
    ret1, ret2 = cm_evaluation(cm)
    return ret1, ret2


def cm_evaluation(cm):
    number_of_class = 3
    sensitivity = []  # TPR
    specificity = []  # TNR=1-FPR
    for c in range(number_of_class):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(number_of_class):
            for j in range(number_of_class):
                if i == j and c == i:
                    TP += cm[i][j]
                elif c == i:
                    FN += cm[i][j]
                elif c == j:
                    FP += cm[i][j]
                else:
                    TN += cm[i][j]
        TPR = TP / float((TP + FN))
        FPR = FP / float((FP + TN))
        sensitivity.append(TPR)
        specificity.append(1 - FPR)
    return np.mean(sensitivity), np.mean(specificity)
