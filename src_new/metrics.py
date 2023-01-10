def recall_at_k(true_list, predicted_list, k=40):
    """
      Calculate the recall at k for a given list of true items and a list of predicted items.

      Parameters:
      - true_list (list): a list of true items.
      - predicted_list (list): a list of predicted items.
      - k (int, optional): the number of items to consider in the predicted list (default is 40).

      Returns:
      - float: the recall at k, i.e., the fraction of true items that appear in the top k predicted items.

      Note:
      The recall at k is a measure of the effectiveness of a recommendation system. It represents the
      proportion of the items that the system recommended (i.e., the top k predicted items) that the user
      actually consumed (i.e., the true items).
      """
    # Sort the predicted list in decreasing order of confidence scores
    sorted_pred = sorted(predicted_list)
    pred = sorted_pred[:k]

    # Convert the true list to a set for fast membership checking
    true_set = set(true_list)

    # Use set intersection to calculate the intersection of the two lists
    inter = true_set & set(pred)

    return round(len(inter) / len(true_list), 3)


def recall_at_k(y_true, y_pred, k=40):
    """
      Calculate the recall at k for a given list of true items and a list of predicted items.

      Parameters:
      - y_true (list): a list of true items.
      - y_pred (list): a list of predicted items.
      - k (int, optional): the number of items to consider in the predicted list (default is 40).

      Returns:
      - float: the recall at k, i.e., the fraction of true items that appear in the top k predicted items.
      """
    # Sort the predicted list in decreasing order of confidence scores
    sorted_pred = sorted(y_pred)
    k_pred = sorted_pred[:k]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(k_pred)

    return round(len(inter) / len(y_true), 3)


def precision_at_k(y_true, y_pred, k=40):
    """
    Calculates the precision at k for the given lists of true and predicted items.
    Precision at k is the fraction of recommended items in the top-k list that are relevant.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The precision at k, rounded to 3 decimal places.
    """

    # Get the top-k predicted items
    pred = y_pred[:k]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(pred)

    # Calculate and return the precision at k
    return round(len(inter) / k, 3)


def r_precision(y_true, y_pred):
    """
    Calculates the R-precision for the given lists of true and predicted items.
    R-precision is the fraction of relevant items in the top-R list that are actually relevant.
    R is the total number of relevant items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.

    Returns:
    float: The R-precision, rounded to 3 decimal places.
    """

    # Get the top-R predicted items, where R is the number of true/relevant items
    R = len(y_true)
    y_pred = y_pred[:R]

    # Use set intersection to calculate the intersection of the two lists
    inter = set(y_true) & set(y_pred)
    # Calculate and return the R-precision
    return round(len(inter) / R, 3)


def reciprocal_rank_at_k(y_true, y_pred, k=40):
    """
    Calculates the reciprocal rank at k for the given lists of true and predicted items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The reciprocal rank at k, or 0 if no relevant items are found in the top-k list.
    """

    # Find the rank of the first relevant item in the top-k list
    flag = False
    for i in range(min(k, len(y_pred))):
        if y_pred[i] in y_true:
            k = i + 1
            flag = True
            break

    # If no relevant items are found, return 0
    if not flag:
        return 0

    # Calculate and return the reciprocal rank at k
    return 1 / k


def f_score(y_true, y_pred, k=40):
    """
    Calculates the F-score for the given lists of true and predicted items.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_pred (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The F-score, or 0 if either precision or recall is 0.
    """
    # Calculate precision and recall
    precision = precision_at_k(y_true, y_pred, k)
    recall = recall_at_k(y_true, y_pred, k)

    # If either precision or recall is 0, return 0
    if precision + recall == 0:
        return 0

    # Calculate and return the F-score
    return (2 * recall * precision) / (recall + precision)


def average_precision(y_true, y_prod, k=40):
    """
    Calculates the average precision for the given lists of true and predicted items.
    The average precision is the average of the precision scores at each relevant item.

    Parameters:
    y_true (list): A list of true/relevant items.
    y_prod (list): A list of predicted items, sorted in order of decreasing relevance.
    k (int, optional): The number of items to consider in the top-k list. Default is 40.

    Returns:
    float: The average precision, or 0 if no relevant items are found in the top-k list.
    """
    # Get the top-k predicted items and initialize variable
    pred = y_prod[:k]
    relevant = 0
    total = 0

    # Iterate over the top-k predicted items
    for i in range(min(k, len(pred))):
        # If the current item is relevant, increment the relevant item count and add the precision score
        if pred[i] in y_true:
            relevant += 1
            total += relevant / (i + 1)

    # If no relevant items are found, return 0
    if relevant == 0:
        return 0
    # Calculate and return the Average Precision
    return round(total / relevant, 3)


def evaluate_all_metrics(y_true, y_pred, k, print_scores=True):
    """
    Evaluates the given y_pred using various metrics.

    Parameters:
    y_true (list): A list of lists of ground truth documents for each query
    y_pred (list): A list of lists of predicted documents for each query
    k (int): The rank at which to compute the metrics
    print_scores (bool, optional): Whether to print the scores for each metric. Default is True.

    Returns:
    dict: A dictionary mapping from metric names to lists of scores for each query
    """
    metrices = {
        'recall@k': recall_at_k,
        'precision@k': precision_at_k,
        'f_score@k': f_score,
        'r-precision': r_precision,
        'MRR@k': reciprocal_rank_at_k,
        'MAP@k': average_precision,
    }

    scores = {name: [] for name in metrices}

    for ground_true, predicted in zip(y_true, y_pred):
        for name, metric in metrices.items():
            scores[name].append(metric(ground_true, predicted, k=k))

    if print_scores:
        for name, values in scores.items():
            print(name, sum(values) / len(values))

    return scores
