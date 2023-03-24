import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dim', type=int, default=108,
                        help="")
    parser.add_argument('--hidden_dim', type=int, default=300,
                        help="")
    parser.add_argument('--epoch', type=int, default=30000,
                        help="")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="")
    parser.add_argument('--device', type=str, default="cuda:1",
                       help="")
    parser.add_argument('--lr', type=float, default=0.0005,
                       help="")
    parser.add_argument('--load_from_disk', type=bool, default=False,
                        help="")
    parser.add_argument('--fairness_notion', type=str, default="DP",
                        help="")
    parser.add_argument('--type_of_abstain', type=str, default="prob",
                        help="clas")
    parser.add_argument('--attribute', type=str, default="sex",
                        help="")

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args