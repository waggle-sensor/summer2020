import argparse
from Classifiers import TextureTemporalClassifier


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-i', 'input', required=True,
                       help='Input video or frame folder to give to the classifier')
    parse.add_argument('-o', 'output', required=True,
                       help='Output folder which will hold water masks')
    args = parse.parse_args()

