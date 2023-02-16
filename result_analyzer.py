
import fileinput
import pandas as pd

def inv_analyzer():
    pass
    

def log_analyzer():
    # read csv table from log/compare_with_ic3ref.csv
    # convert to pandas dataframe
    df = pd.read_csv('log/compare_with_ic3ref.csv')
    # remove the duplicated rows with same case name
    df.drop_duplicates(subset=['case name'], keep='last', inplace=True)
    print(df)
    
    

if __name__ == "__main__":
    # parser.add_argument('--compare_inv', action='store_true', help='compare the inv with ic3ref')
    # parser.add_argument('--compare_log', action='store_true', help='compare the log with ic3ref')
    # parser.add_argument('--aig-case-folder-prefix-for-prediction', type=str, default=None, help='case folder, use for test all cases in the folder, for example: case4test/hwmcc2007')
    # parser.add_argument('--aig-case-folder-prefix-for-ic3ref', type=str, default=None, help='case folder, contains all ic3ref produced inv.cnf, for example: case4test/hwmcc2007')
    # args = parser.parse_args()
    log_analyzer()