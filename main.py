import argparse
from lc.database import create_db

def main():
    parser = argparse.ArgumentParser()

    # options = 'last_mile', 'two_sided', 'new'
    create_db('new')

if __name__ == '__main__':
    main()
