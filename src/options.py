import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epoch', type=int, default=30000,
                        help="Maximum Epochs")
    parser.add_argument('--stopping_ratio', type=float, default=0.005, help="Early Stopping for Classifiers Training.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="")
    parser.add_argument('--device', type=str, default="cuda:1",
                       help="")
    parser.add_argument('--lr', type=float, default=0.0005,
                       help="learning rate")
    # parser.add_argument('--load_from_disk', type=bool, default=False,
    #                     help="")
    parser.add_argument('--sample_ratio', type=float, default=0.5,
                        help="sample how many data for IP.")
    parser.add_argument('--seed', type=int, default=1, help='random seed')


    parser.add_argument('--dataset', type=str, default="adult",
                        help="Dataset")
    parser.add_argument('--attribute', type=str, default="sex",
                        help="sensitive attribute")
    parser.add_argument('--fairness_notion', type=str, default="DP",
                        help="")
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help="disparity")
    parser.add_argument('--delta1', type=float, default=0.2,
                        help="abstention rate of group1")
    parser.add_argument('--delta2', type=float, default=0.2,
                        help="abstention rate of group2")
    parser.add_argument('--sigma1', type=float, default=1,
                        help="abstention rate disparity of positive samples")
    parser.add_argument('--sigma0', type=float, default=1,
                        help="abstention rate disparity of negative samples")
    parser.add_argument('--sigma', type=float, default=1,
                        help="abstention rate disparity of all samples")
    parser.add_argument('--eta1', type=float, default=0,
                        help="error relaxation rate of group1")
    parser.add_argument('--eta2', type=float, default=0,
                        help="error relaxation rate of group2")

    args = parser.parse_args()
    return args