import argparse
from lc.database import create_db
from lc.chain import wizard_chain, t5_chain, incite_chain


def main():
    parser = argparse.ArgumentParser()

    create_db()
    # parser.add_argument(
    #     '--data', dest='data',
    #     action = 'store_true',
    #     help="Use this argument to generate the databse from the PDFs"
    # )

    # parser.add_argument(
    #     '--chain', dest='chain',
    #     action = 'store_true',
    #     help="Use this argument to run the chain"
    # )

    # args = parser.parse_args()

    # if args.data:
    #     create_db()

    # if args.chain:
    #     #wizard_chain()
    #     answer = t5_chain()
    #     #answer = incite_chain(query = "Name five main topics discussed in the studies ")

    #     print(answer)

if __name__ == '__main__':
    main()
