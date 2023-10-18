import pandas as pd
from lc.database import create_db, create_multiple_db

from lc.chain import DefineChain

def main():
    # options = 'last_mile', 'two_sided', 'new'
    # create_db('new')

    #Create Individual Databases for each PDF in the literature
    # options = 'transporation'
    inp = 'transporation'
    create_multiple_db('transportation')

    # sum_chain = DefineChain('transportation')
    # summary = sum_chain.individual_chain()
    # summary_df = pd.DataFrame(summary)
    # summary_df.to_csv('summaries_' + inp + '.csv')
    # print(summary)

if __name__ == '__main__':
    main()
