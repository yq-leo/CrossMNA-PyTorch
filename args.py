from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='phone-email',
                        choices=['phone-email', 'ACM-DBLP', 'foursquare-twitter'],
                        help='available datasets: phone-email, ACM-DBLP, foursquare-twitter')
    parser.add_argument("--p", default=0.2, type=str)
    parser.add_argument('--node-dim', default=200, type=int, help='d1')
    parser.add_argument('--layer-dim', default=100, type=int, help='d2')
    parser.add_argument('--batch-size', default=512 * 8, type=int)
    parser.add_argument('--neg-samples', default=1, type=int)
    parser.add_argument('--epochs', default=400, type=int)

    return parser.parse_args()
