
import fileinput
import pandas as pd

def inv_analyzer():
    pass
    

def log_analyzer():
    # read csv table from log/compare_with_ic3ref.csv
    # convert to pandas dataframe
    df = pd.read_csv('log/compare_with_ic3ref.csv')
    # only keep the row that  NN-IC3-bg, IC3ref-bg are both 1
    df = df[(df[" NN-IC3-bg"] == 1) & (df[" IC3ref-bg"] == 1) | (df[" NN-IC3-bg"] == 0) & (df[" IC3ref-bg"] == 0)]
    # sort the dataframe by case name and the "NN-IC3 Frame" column
    df = df.sort_values(by=["case name", " NN-IC3 Time"])
    # remove the duplicated rows
    df.drop_duplicates(subset="case name", keep="first", inplace=True)
    # re-index the dataframe
    df = df.reset_index(drop=True)
    #print(df)
    # only print the table with top 3 columns
    print(df.iloc[:, :-2])
    
    

if __name__ == "__main__":
    # parser.add_argument('--compare_inv', action='store_true', help='compare the inv with ic3ref')
    # parser.add_argument('--compare_log', action='store_true', help='compare the log with ic3ref')
    # parser.add_argument('--aig-case-folder-prefix-for-prediction', type=str, default=None, help='case folder, use for test all cases in the folder, for example: benchmark_folder/hwmcc2007')
    # parser.add_argument('--aig-case-folder-prefix-for-ic3ref', type=str, default=None, help='case folder, contains all ic3ref produced inv.cnf, for example: benchmark_folder/hwmcc2007')
    # args = parser.parse_args()
    log_analyzer()