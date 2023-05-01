import numpy as np
import os
import time
from scipy.optimize import milp, Bounds
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint

def IP(data, args, OptimalNet):
    X = data.X.numpy()
    y = data.y.numpy().astype(int)
    group1_index = args.group_indices[0]
    group2_index = args.group_indices[1]

    pred_labels = ((OptimalNet(data.X) >= 0.5) * 1).numpy().flatten()

    n = X.shape[0]
    g2_indices = np.where(X[:, group2_index] == 1)[0]
    g1_indices = np.where(X[:, group1_index] == 1)[0]

    g1_v = (X[:, group1_index] == 1) * 1
    g2_v = (X[:, group2_index] == 1) * 1

    g1_num = g1_indices.shape[0]
    g2_num = g2_indices.shape[0]
    total_error = pred_labels != y
    g1_error = (pred_labels != y)[g1_indices]
    g2_error = (pred_labels != y)[g2_indices]
    g1_error_rate = np.sum(g1_error) / g1_num
    g2_error_rate = np.sum(g2_error) / g2_num
    total_error_rate = np.sum(total_error) / n

    zero_vector = np.zeros_like(y)
    zero_matrix = np.zeros((n, n))

    a = np.hstack([1 - 2 * y, y, zero_vector])

    if args.fairness_notion == "DP":
        c1 = np.stack(
            [
                np.hstack([g1_v / g1_num - g2_v / g2_num, np.zeros(2 * n)]),
                np.hstack([g2_v / g2_num - g1_v / g1_num, np.zeros(2 * n)]),
            ]
        )
        #     print("c1", c1.shape)
        cons1 = args.epsilon * np.ones(2)

    elif args.fairness_notion == "EO":
        temp1 = (y * g1_v)
        temp2 = (y * g2_v)

        c1 = np.stack([
            np.hstack([temp1 * np.sum(temp2) - temp2 * np.sum(temp1), np.zeros(2 * n)]),
            np.hstack([temp2 * np.sum(temp1) - temp1 * np.sum(temp2), np.zeros(2 * n)])])
        # print("c1", c1.shape)
        cons1 = np.array([args.epsilon * np.sum(y * g1_v) * np.sum(y * g2_v),
                          args.epsilon * np.sum(y * g1_v) * np.sum(y * g2_v)])
        # print("cons1", cons1.shape)

    elif args.fairness_notion == "EOs":
        temp1 = (y * g1_v)
        temp2 = (y * g2_v)
        temp3 = ((1 - y) * g1_v)
        temp4 = ((1 - y) * g2_v)

        c1 = np.stack([
            np.hstack([temp1 * np.sum(temp2) - temp2 * np.sum(temp1), np.zeros(2 * n)]),
            np.hstack([temp2 * np.sum(temp1) - temp1 * np.sum(temp2), np.zeros(2 * n)]),
            np.hstack([temp3 * np.sum(temp4) - temp4 * np.sum(temp3), np.zeros(2 * n)]),
            np.hstack([temp4 * np.sum(temp3) - temp3 * np.sum(temp4), np.zeros(2 * n)])
        ])
        # print("c1", c1.shape)
        cons1 = np.array(
            [
                args.epsilon * np.sum(y * g1_v) * np.sum(y * g2_v),
                args.epsilon * np.sum(y * g1_v) * np.sum(y * g2_v),
                args.epsilon * np.sum((1 - y) * g1_v) * np.sum((1 - y) * g2_v),
                args.epsilon * np.sum((1 - y) * g1_v) * np.sum((1 - y) * g2_v),
            ]
        )
        # print("cons1", cons1.shape)


    c2 = np.stack(
        [
            np.hstack([zero_vector, g1_v, zero_vector]),
            np.hstack([zero_vector, g2_v, zero_vector]),
        ]
    )
    #     print("c2", c2.shape)
    cons2 = np.array([(1 - args.delta1) * g1_num, (1 - args.delta2) * g2_num])

    g1_v - 2 * g1_v * y
    r1 = (g1_error_rate + args.eta1 - args.eta1 * g1_error_rate) * g1_v - g1_v * y
    r2 = (g2_error_rate + args.eta2 - args.eta2 * g2_error_rate) * g2_v - g2_v * y
    c3 = np.stack([
        np.hstack([g1_v - 2 * g1_v * y, -r1, zero_vector]),
        np.hstack([g2_v - 2 * g2_v * y, -r2, zero_vector]),
    ])
    cons3 = np.zeros(2)

    I = np.eye(n)
    c4 = np.concatenate([I, -I, zero_matrix], axis=1)
    cons4 = np.zeros(n)

    c5 = np.concatenate([I, zero_matrix, -I], axis=1)
    cons5 = cons4

    c6 = np.concatenate([-I, I, I], axis=1)
    cons6 = np.ones_like(y)

    c7 = np.stack(
        [
            np.hstack([np.zeros(n), g1_v * y / np.sum(g1_v * y) - g2_v * y / np.sum(g2_v * y), np.zeros(n)]),
            np.hstack([np.zeros(n), g2_v * y / np.sum(g2_v * y) - g1_v * y / np.sum(g1_v * y), np.zeros(n)]),
            np.hstack(
                [np.zeros(n), g1_v * (1 - y) / np.sum(g1_v * (1 - y)) - g2_v * (1 - y) / np.sum(g2_v * (1 - y)),
                 np.zeros(n)]),
            np.hstack(
                [np.zeros(n), g2_v * (1 - y) / np.sum(g2_v * (1 - y)) - g1_v * (1 - y) / np.sum(g1_v * (1 - y)),
                 np.zeros(n)]),
            np.hstack([np.zeros(n), g1_v / g1_num - g2_v / g2_num, np.zeros(n)]),
            np.hstack([np.zeros(n), g2_v / g2_num - g1_v / g1_num, np.zeros(n)]),
        ]
    )
    cons7 = np.array([args.sigma1, args.sigma1, args.sigma0, args.sigma0, args.sigma, args.sigma])

    lower_bounds = np.zeros_like(a)
    upper_bounds = np.ones_like(a)

    constraints = [
        LinearConstraint(c1, ub=cons1),
        LinearConstraint(c2, lb=cons2),
        LinearConstraint(c3, ub=cons3),
        LinearConstraint(c4, ub=cons4),
        LinearConstraint(c5, ub=cons5),
        LinearConstraint(c6, ub=cons6),
        LinearConstraint(c7, ub=cons7),
    ]
    integrality = np.ones_like(a)
    res = milp(
        c=a,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds)
    )
    kn = res.x
    if kn is None:
        print("Not feasible!")
        return None, None
    num_kn = kn.shape[0]
    interval = int(num_kn / 3)
    wn = kn[interval:2 * interval]
    hn = kn[2 * interval:]
    return wn, hn


def process_wn(wn, pred_labels):
    if wn is None:
        print("Not feasible!")
        return None
    else:
        plt.plot(np.sort(wn))
        plt.title("$w_n$ for each data point")
        plt.xlabel("data points")
        plt.ylabel("value")

        return (wn * pred_labels)