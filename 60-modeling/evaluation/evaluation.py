"""Evaluate logits and labels"""
def evaluate(logits, labels):
    tp = ((logits > 0) & (labels == 1)).sum().item()
    tn = ((logits < 0) & (labels == 0)).sum().item()
    fp = ((logits > 0) & (labels == 0)).sum().item()
    fn = ((logits < 0) & (labels == 1)).sum().item()
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return 100*accuracy, 100*precision, 100*recall, 100*f1