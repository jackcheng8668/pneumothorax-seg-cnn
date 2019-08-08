from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
keras = tf.keras

class BinaryMetricsCallback(keras.callbacks.Callback):
    """
    Keras callback to calculate metrics of a binary classifier for each epoch.
    Attributes:
        val_data:
            The validation data.
            Either:
                a Sequence object for the validation data
                tuple (x_val, y_val)
        batch_size: batch size. Defaults to None, which in turn, defaults to 32 for tf.keras
    """
    def __init__(self, val_data, batch_size=None):
        super().__init__()
        self.val_data = val_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        if isinstance(self.val_data, (tuple, list)):
            pred = self.model.predict(self.val_data[0], batch_size=self.batch_size)
            # simulates a 0.5 threshold
            round_predict_results = np.rint(pred)
            tn, fp, fn, tp = confusion_matrix(self.val_data[1].ravel(), pred.ravel()).ravel()
            logs["val_precision"], logs["val_recall"], logs["val_f1"] = self.evaluate(tn, fp, fn)
        else: # generators/sequence
            tp, fp, fn = 0, 0, 0
            try:
                # iterates and store the number of true positives, false positives and false negatives
                for i in range(len(self.val_data)):
                    x_batch, y_batch = self.val_data[i]
                    pred_batch = self.model.predict(x_batch, batch_size=self.batch_size)
                    _, fp_batch, fn_batch, tp_batch = confusion_matrix(y_batch.ravel(), pred.ravel()).ravel()
                    tp+=tp_batch
                    fp+=fp_batch
                    fn+=tn_batch
                logs["val_precision"], logs["val_recall"], logs["val_f1"] = self.evaluate(tn, fp, fn)
            except:
                print("Please make sure that val_data is either a generator or an instance of keras.utils.Sequence")
        def round_results(metric):
            logs[metric] = round(logs[metric], 4)
            return
        _ = map(round_results, list(logs.keys()))
        print(" - val_precision: {} - val_recall: {} - val_f1: {}".format(logs["precision"], logs["recall"], logs["f1"]))

    def evaluate(self, tp, fp, fn):
        """
        Calculates relevant binary metrics. Includes: precision, recall, f1-score
        Args:
            tp: number of true positives
            fp: number of false positives
            fn: number of false negatives
        Returns:
            precision, reall, f1-score
        """
        ppv, tpr = precision(tp, fp), recall(tp, fn)
        f1 = f1score(ppv, tpr)
        return ppv, tpr, f1

def f1score(precision_value, recall_value, epsilon=1e-5):
    """
    Calculating F1-score from precision and recall to reduce computation redundancy.
    Args:
        precision_value: precision (0-1)
        recall_value: recall (0-1)
    Returns:
        F1 score (0-1)
    """
    return 2 * (precision_value * recall_value) / (precision_value + recall_value + epsilon)

def precision(tp, fp, epsilon=1e-5):
    """
    Calculates precision (a.k.a. positive predictive value) for binary classification.
    Args:
        tp: number of true positives
        fp: number of false positives
    Returns:
        precision value (0-1)
    """
    return tp / (tp + fp + epsilon)

def recall(tp, fn, epsilon=1e-5):
    """
    Calculates recall (a.k.a. true positive rate) for binary classification/segmentation
    Args:
        tp: number of true positives
        fn: number of false negatives
    Returns:
        recall value (0-1)
    """
    return tp / (tp + fn + epsilon)
