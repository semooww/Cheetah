import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np


# Display the distribution of data on the train and test
def plot_pie_sets(arrays, technique):
    titles = ["Train Set", "Test Set"]
    labels = ["Normal", "Cataract", "Glaucoma", "Retina Disease"]
    fig = plt.figure(figsize=(12, 5))
    plt.title(f"Distribution of {technique}")
    plt.axis('off')
    plt.grid(False)
    for i in range(2):
        fig.add_subplot(1, 2, i + 1)
        arr = []
        for j in range(4):
            arr.append((arrays[i] == j).sum())
        plt.title(titles[i])
        plt.pie(arr, autopct=lambda x: '{:.0f}'.format(x * np.array(arr).sum() / 100))
        plt.legend(labels=labels, loc=0)
    plt.savefig(f"Plots/{technique}_pie.png")


def plot_history(arrays, technique):
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
    plt.savefig(f"Plots/{technique}_history.png")


def plot_conf_matrix(model, X_test, y_test, technique):
    labels = ["Normal", "Cataract", "Glaucoma", "Retina Disease"]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.grid(False)
    plt.title(f"Confusion Matrix of {technique}")
    plt.savefig(f"Plots/{technique}_confusion_matrix.png")
    return cm_evaluation(cm)


def cm_evaluation(cm):
    number_of_class = 4
    sensitivity = np.zeros(number_of_class, dtype="float32")  # TPR
    specificity = np.zeros(number_of_class, dtype="float32")  # TNR=1-FPR
    for c in range(number_of_class):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(cm[0])):
            for j in range(len(cm[[0]])):
                if i == j and c == i:
                    TP += cm[i][j]
                elif c == i:
                    FN += cm[i][j]
                elif c == j:
                    FP += cm[i][j]
                else:
                    TN += cm[i][j]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        sensitivity[c] = TPR
        specificity[c] = 1 - FPR
    return np.average(sensitivity), np.average(specificity)
