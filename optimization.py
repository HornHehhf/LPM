import numpy as np
import cvxpy as cp
import csv
from cvxpy import log_sum_exp, sum, Minimize, Problem
from scipy import spatial


def cross_entropy_loss(logits, label):
    return -logits[label] + log_sum_exp(logits)
    # log_sum_exp accuracy


def get_total_loss(X, c, t1, t2, R):
    logits = X[0, c:]
    label = 0
    total_loss = cross_entropy_loss(logits, label)
    for i in range(t1)[1:]:
        logits = X[i, c:]
        label = i
        total_loss += cross_entropy_loss(logits, label)
    if R != float('inf'):
        for j in range(t2):
            logits = X[t1 + j, c:]
            label = t1 + j
            total_loss += (cross_entropy_loss(logits, label) / R)
    return total_loss


def get_constraints(X, c, Af, Aw):
    constraints = []
    constraints += [sum([X[i, i] for i in range(c)]) <= c * Af]
    constraints += [sum([X[c + j, c + j] for j in range(c)]) <= c * Aw]
    constraints += [X >> 0]
    # X >= 0
    return constraints


def neural_collapse_optimization(class_num, big_class_num, ratio, feature_constant, weight_constant):
    c = class_num
    t1 = big_class_num
    t2 = c - t1
    R = ratio
    Af = feature_constant
    Aw = weight_constant
    X = cp.Variable((2 * c, 2 * c), symmetric=True)
    total_loss = get_total_loss(X, c, t1, t2, R)
    constraints = get_constraints(X, c, Af, Aw)
    obj = Minimize(total_loss)
    prob = Problem(obj, constraints)
    try:
        prob.solve()
    except Exception as e:
        print(e)
    '''
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(X.value)
    print("A dual solution corresponding to the inequality constraints is")
    print(prob.constraints[0].dual_value)
    '''
    X_round = []
    for i in range(len(X.value)):
        X_round.append([round(X.value[i][j], 3) for j in range(len(X.value[0]))])
    # print(X_round)
    with open('tmp_matrix.csv', 'w') as f:
        writer = csv.writer(f)
        for i in range(len(X_round)):
            writer.writerow(X_round[i])
    between_class_cos_small = []
    for i in range(2 * c)[c + t1: ]:
        for j in range(2 * c)[c + t1:]:
            if i != j:
                cos_value = X.value[i, j] / np.sqrt(X.value[i, i] * X.value[j, j])
                between_class_cos_small.append(cos_value)
    print(X.value[c + t1: 2 * c, c + t1: 2 * c])
    print('avg between-class weight cosine for small classes', np.mean(between_class_cos_small))
    print('std between-class weight cosine for small classes', np.std(between_class_cos_small))
    return np.mean(between_class_cos_small)


def run_optimization_experiments():
    class_num = 10
    big_class_num = 5
    feature_constant = 5
    weight_constant = 1
    ratio_list = [np.power(10, i * 0.1) for i in range(41)]
    cos_values = []
    for x in range(len(ratio_list)):
        ratio = ratio_list[x]
        print('ratio=', ratio)
        cos_values.append(neural_collapse_optimization(class_num, big_class_num, ratio,
                                                       feature_constant, weight_constant))

    print('rounded cos values', [round(cos_value, 2) for cos_value in cos_values])


if __name__ == '__main__':
    run_optimization_experiments()
