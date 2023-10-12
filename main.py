import argparse
from lc.database import create_db

from lc.chain import DefineChain

def main():
    parser = argparse.ArgumentParser()

    # options = 'last_mile', 'two_sided', 'new'
    # create_db('new')

    sum_chain = DefineChain('new')
    summary = sum_chain.summarizer_chain()
    print(summary)

if __name__ == '__main__':
    main()
