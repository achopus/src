import torch
from torch import Tensor
from torch.nn.functional import sigmoid
import matplotlib.pyplot as plt

positive_class = 1
negative_class = 0

class Metrics:
    def __init__(self) -> None:
        self.classifier_outputs: Tensor = None
        self.predictions: Tensor = None
        self.labels: Tensor = None

        self.P: float = None
        self.N: float = None
        self.T: float = None

        self.TP: float = None
        self.TN: float = None
        self.FP: float = None
        self.FN: float = None

        self.ROC: Tensor = None
        self.AUC: float = None

        self.precision: float = None
        self.recall: float = None
        self.accuracy: float = None

    def reset(self) -> None:
        self.__init__()

    def set_values(self, classifier_outputs: Tensor, labels: Tensor, output_mode: str = 'logits') -> None:
        """
        Set values from the classifier model and ground-truth labels.
        Args:
            classifier_outputs (Tensor [N]): Prediction of the model, either logits or probabilities
            labels (Tensor [N]): Numeric ground-truth labels of data (positive classes has biggest value)
            output_mode (str): Determin whether classifiter_outputs are probabilites or logits. (available options = ['logits', 'probability'])
        """

        if len(classifier_outputs) != len(labels):
            raise AssertionError(f"Length of classifier outputs ({len(classifier_outputs)}) isn't equal to length of the labels ({len(labels)}).")
        
        classes = torch.unique(labels)
        if len(classes) != 2:
            raise AssertionError(f"Labels contain more than two classes! ({len(classes)})")

        if output_mode not in ['probability', 'logits']:
            raise AssertionError("output_mode must be either 'probability' or 'logits'.")

        # Use CPU for simplicity
        classifier_outputs = classifier_outputs.cpu()
        labels = labels.cpu()

        # Set labels to '1' for positive class and '0' for negative class
        labels_c = torch.zeros_like(labels)
        labels_c[labels == labels.max()] = 1

        # Convert classifier outputs to predictions (If outputs are already probabilites, just get classes
        if output_mode == 'logits':
            predictions = (sigmoid(classifier_outputs) > 0.5).int()
        else:
            predictions = (classifier_outputs > 0.5).int()

        # Set values
        self.classifier_outputs = classifier_outputs
        self.predictions = predictions
        self.labels = labels

        self.get_basic_metrics()

    def get_basic_metrics(self) -> None:
        """
        Compute basic statistical metrics like accuracy, precision, recall amongst others.
        """
        labels = self.labels
        predictions = self.predictions

        self.T = len(labels)
        self.P = torch.sum(labels == positive_class)
        self.N = torch.sum(labels != positive_class)

        self.TP = torch.sum(torch.logical_and(predictions == positive_class, labels == positive_class))
        self.FP = torch.sum(torch.logical_and(predictions == positive_class, labels != positive_class))
        self.TN = torch.sum(torch.logical_and(predictions == positive_class, labels != positive_class))
        self.FN = torch.sum(torch.logical_and(predictions != positive_class, labels == positive_class))

        alpha = (self.P / self.N).item()

        self.accuracy = torch.mean((predictions == labels).float())
        self.precision = self.TP / (self.TP + self.FP)
        self.precision_weighted = self.TP / (self.TP + alpha * self.FP)
        self.recall = self.TP / (self.TP + self.FN)

        self.F1 = self.get_F_score(beta=1.0)


    def get_F_score(self, beta: float) -> float:
        return (1 + beta**2) * (self.precision * self.recall) / (beta**2 * self.precision + self.recall)

    def get_AUC_and_ROC(self) -> tuple[float, Tensor]:
        """
        Computes ROC and AUC of given binary classifier.

        Returns:
            AUC (float [1]): Area under the curve.
            ROC (Tensor [2, N]): Receiver operator characteristics curve. (TPR, FPR) 
        """

        # Reorder ouputs from biggest to smallest
        order = torch.argsort(self.classifier_outputs, descending=True)
        labels = self.labels[order]
        
        # Compute ROC curve
        TPR = torch.cumsum(labels == positive_class, dim=0)
        FPR = torch.cumsum(labels != positive_class, dim=0)
        TPR = TPR / TPR[-1]
        FPR = FPR / FPR[-1]

        # Get AUC by the trapezoid rule
        AUC = torch.trapezoid(TPR, FPR).item()
        ROC = torch.stack([TPR, FPR])

        self.AUC = AUC
        self.ROC = ROC
        return AUC, ROC
    
    def show_ROC(self) -> None:
        auc = self.AUC
        TPR = self.ROC[0].numpy()
        FPR = self.ROC[1].numpy()

        plt.figure('ROC')
        plt.title(f"Receive operator characteric curve (AUC = {round(auc, 2)})", weight='bold', fontsize=20)
        plt.plot(FPR, TPR, color='blue', linewidth=2)
        plt.fill_between(FPR, TPR, alpha=0.75)
        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('False Positive Rate', weight='bold', fontsize=16)
        plt.ylabel('True Positive Rate', weight='bold', fontsize=16)
        plt.box(False)
        plt.grid(linestyle='dashed', alpha=0.5)
        plt.show()    


    def __str__(self) -> str:
        out = ""
        out += f"Metrics of the predictions\n-------------------\n"
        out += f"Number of predictions: {len(self.predictions)}\n"
        out += f"Accuracy: {torch.round(100 * self.accuracy)}%\n"
        out += f"Precision: {round(self.precision.item(), 3)} | Precision weighted: {round(self.precision_weighted.item(), 3)}\n"
        out += f"Recall: {self.recall}\n"
        out += f"F1 score: {round(self.F1.item(), 3)}\n"
        
        if not self.AUC: return out
        out += f"AUC: {round(self.AUC, 3)}"
        self.show_ROC()
        
        return out 
