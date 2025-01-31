import numpy as np

class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: prediction
        # axis = 1: target
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        hist[label_pred, label_true] += 1

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.item(), lp.item(), self.n_classes)    #lt.item(), lp.item()
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along

        if sum(hist.sum(axis=1)) != 0:
            acc = sum(np.diag(hist)) / sum(hist.sum(axis=1))
        else:
            acc = 0.0
        
        return acc

    def calculate_precision_recall(self):
        """Calculate precision and recall for each class"""
        hist = self.confusion_matrix
        precision = np.zeros(self.n_classes)
        recall = np.zeros (self.n_classes)

        for i in range(self.n_classes):
            tp = hist[i, i]
            fp = np.sum(hist[:, i]) -tp
            fn = np.sum(hist[i, :]) -tp

            precision[i] = tp / (tp+fp) if tp + fp > 0 else 0
            recall[i] = tp /(tp +fn) if tp + fn > 0 else 0

        return precision, recall
    
    def plotcm(self):
        print(self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
