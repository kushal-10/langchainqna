import argparse
from lc.database import create_db
from lc.chain import wizard_chain, t5_chain, incite_chain, t5_chain_new, t5_llm_chain


def main():
    parser = argparse.ArgumentParser()

    # options = 'last_mile', 'two_sided', 'new'
    create_db('last_mile')

    # answer, source = t5_chain_new(query = "what is the unique contribution of this study?")
    # answer, source = t5_chain_new(query = "what is a P2P rental platform?")

    # ans = t5_llm_chain(query = "what is a P2P rental platform?")
    # ans = t5_llm_chain(query = "what is the unique contribution of this study?")
    # print(ans)
    # print(len(ans))


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
