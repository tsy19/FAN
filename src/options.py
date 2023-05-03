import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate the model that already trained')
    parser.add_argument('--max_epoch', type=int, default=500,
                        help="Maximum epochs allowed if early stopping not happen.")
    parser.add_argument('--min_epoch', type=int, default=0,
                        help="Minimum epochs allowed for early stopping.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early stopping patience. The model will stop training if the loss not decrease for number of patience epochs.")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Training batch size.")
    parser.add_argument('--device', type=str, default="cuda:1",
                       help="Training device.")
    parser.add_argument('--lr', type=float, default=0.001,
                       help="Initial learning rate.")
    parser.add_argument('--lr_factor', type=float, default=0.95,
                        help="new learning rate = previous learning rate * lr_factor")
    parser.add_argument('--lr_patience', type=int, default=10,
                        help="if the loss not decrease after lr_patience then reduce learning rate.")
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                        help="Sample how much data for Integer Programming.")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--ROOT', type=str, default="./", help='Root to the directory.')


    parser.add_argument('--dataset', type=str, default="adult",
                        help="Dataset. Select from adult, law, compas")
    parser.add_argument('--attribute', type=str, default="sex",
                        help="Sensitive attribute to protect. For adult, default is sex.")
    parser.add_argument('--fairness_notion', type=str, default="DP",
                        help="Fairness notion to use. Default is DP. Choose from DP, EO, EOs.")
    parser.add_argument('--epsilon', type=float, default=0.03,
                        help="A fairness measurement. Disparity of the two groups.")
    parser.add_argument('--delta1', type=float, default=0.2,
                        help="Abstention rate allowed of group1")
    parser.add_argument('--delta2', type=float, default=0.2,
                        help="Abstention rate allowed of group2")
    parser.add_argument('--sigma1', type=float, default=1.0,
                        help="Abstention rate disparity allowed for two groups within positive samples. Default 1 means no restriction.")
    parser.add_argument('--sigma0', type=float, default=1.0,
                        help="Abstention rate disparity allowed for two groups within negative samples. Default 1 means no restriction.")
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="Abstention rate disparity allowed for two groups. Default 1 means no restriction.")
    parser.add_argument('--eta1', type=float, default=0,
                        help="Error relaxation rate of group1.")
    parser.add_argument('--eta2', type=float, default=0,
                        help="Error relaxation rate of group2.")

    args = parser.parse_args()
    return args
