"""
Has the functions to evaluate the Top 5 and Top 1 cases

Authors: Hojin Kang and Tomas Nunez
"""


def eval_top5(pred_reales, top5_obtenido):
    """
    Function that evaluates Accuracy for Top 5

    :param pred_reales:         Real tags for images
    :param top5_obtenido:       The Top 5 predictions obtained for each case
    :return:                    A float corresponding to the Accuracy of the model over the examples
    """
    count = 0.
    total = 0.
    i = 0
    for top5_temp in top5_obtenido:
        pred_real = pred_reales[i]
        if pred_real in top5_temp:
            count = count + 1
        total = total + 1
        i = i + 1
    return count/total*100


# Evaluates for Top 1
def eval_top1(pred_reales, top1_obtenido):
    """
    Function that evaluates Accuracy for Top 1

    :param pred_reales:         Real tags for images
    :param top1_obtenido:       The Top prediction obtained for each case
    :return:                    A float corresponding to the Accuracy of the model over the examples
    """
    count = 0.
    total = 0.
    i = 0
    for top1_temp in top1_obtenido:
        pred_real = pred_reales[i]
        if pred_real == top1_temp:
            count = count + 1
        total = total + 1
        i = i + 1
    return count/total*100
