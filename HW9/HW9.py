import numpy as np
from matplotlib import pyplot as plt
from HodaDataset.HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

# Load Hoda Farsi Digit Dataset
train_images, train_labels = read_hoda_cdb('./HodaDataset/DigitDB/Train 60000.cdb')
test_images, test_labels = read_hoda_cdb('./HodaDataset/DigitDB/Test 20000.cdb')


#
def draw_confusion_matrix(y_true, y_pred, classes=None, normalize=True, title=None, cmap=plt.cm.Blues):
    acc = np.sum(y_true == y_pred) / len(y_true)
    print('Accuracy = {}'.format(acc))

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Confusion Matrix = \n{}'.format(np.round(cm, 3)))

    if classes is None:
        classes = [str(i) for i in range(len(np.unique(y_true)))]

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#
def extract_geometrical_features(images):
    # Modify this code
    features = np.zeros((len(images), 3))
    for i in range(len(images)):
        features[i, 0] = np.sum(images[i])
        features[i, 1] = np.sum(images[i])
        features[i, 2] = np.sum(images[i])

    return features


#
def extract_textural_features(images):
    # Type your code here
    return features


#
train_features = extract_geometrical_features(train_images)
test_features = extract_geometrical_features(test_images)


model1 = LinearSVC().fit(train_features, train_labels)
test_predictions = model1.predict(test_features)

draw_confusion_matrix(test_labels, test_predictions, title='SVM + Geometric')

plt.show()

# Type your code here
