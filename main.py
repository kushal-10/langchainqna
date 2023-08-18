import argparse
from lc.database import create_db
from lc.model import load_model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', dest='data',
        action = 'store_true',
        help="Use this argument to generate the databse from the PDFs"
    )

    parser.add_argument(
        '--model1', dest='model1',
        action = 'store_true',
        help="Use this argument to load the WizardLM Model"
    )

    args = parser.parse_args()

    if args.data:
        create_db()

    if args.model1:
        load_model()

if __name__ == '__main__':
    main()
