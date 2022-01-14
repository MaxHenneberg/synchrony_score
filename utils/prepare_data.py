import glob
import os
import csv
import re
import pandas as pd
from typing import Callable, List

basefolder = '..\\resources'
targetFolder = 'Prepared_Data'


def get_filenames(path: str, pattern: str = "*.csv") -> List[str]:
    """
    Returns a list of filenames in a path.

    Args:
        path (str): Search for *.csv files in this directory.

    Returns:
        filenames (List[str]): list of filenames matching the pattern.
    """
    glob_path = os.path.join(f"{path}", pattern)
    file_paths = glob.glob(glob_path)
    return [e.split(os.path.sep)[-1] for e in file_paths]


def read_data_as_df(folder: str, columns: List[str]) -> (pd.DataFrame, pd.DataFrame):
    '''
    Reads in user study data a df for each user.
    Args:
        folder (str): Contains a csv file per user study
        columns (list(str)): the list of column names to read
    
    Returns:
        (dfs_user_one, dfs_user_two) (dict(pd.DataFrame), dict(pd.DataFrame)):
        a tuple of user one and user two dictionaries keys being the study and
        values of the read in data.
    '''
    dfs_user_one = dict()
    dfs_user_two = dict()
    pathToFolder = os.path.join(basefolder, folder)
    file_names = get_filenames(pathToFolder)  # file_names_list(folder)
    for fname in file_names:
        df = pd.read_csv(os.path.join(pathToFolder, fname), usecols=columns)
        # Find user id in filename: "User Study 1 - User 1 - Intro.csv"      
        p = re.findall(r'\d+', fname)
        study_id = int(p[0])
        user_id = int(p[1])
        if user_id == 1:
            dfs_user_one[study_id] = df
        elif user_id == 2:
            dfs_user_two[study_id] = df
        else:
            raise ValueError(f"User id must be 1 or 2, but is {user_id}.")
    return dfs_user_one, dfs_user_two


def create_excel_study_summary(sourceFolder: str, summaryName: str, columns: List[str],
                               merge_and_clean: Callable) -> None:
    """
    Create excel file summary for the given folder.
    
    The file contains a worksheet per user study.
    
    Args:
        sourceFolder (str): Contains a csv file per user and study.
        summaryName (str): Name of the xls file
        columns (list[str]): Will be present in the summary.
        merge_and_clean (Callable): function that takes two DataFrames - one per user.
            It cleans the data, merges the DataFrame index and returns a merged df of the form:
            frame, timestamp, user_one_datatypeX, user_two_datatypeX
    """
    dfs_user_one, dfs_user_two = read_data_as_df(sourceFolder, columns)
    xls_writer = pd.ExcelWriter(os.path.join(os.path.join(basefolder, targetFolder), f"{summaryName}.xlsx"),
                                engine="xlsxwriter")
    for study_id in sorted(dfs_user_one.keys()):
        print(study_id)
        df_merged = merge_and_clean(dfs_user_one[study_id], dfs_user_two[study_id])
        df_merged.to_excel(xls_writer, sheet_name=f"Study{study_id}")
    xls_writer.save()
    print('Excel Saved')


def load_session_data(file_name: str) -> {pd.DataFrame}:
    df_dict = dict()
    for study in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
        df_dict[study] = pd.read_excel(os.path.join(os.path.join(basefolder, targetFolder), f"{file_name}.xlsx"),
                                       sheet_name=f"Study{study}", index_col=0)
    return df_dict


def collectUserData(excelFile, processUser1, processUser2):
    excel = load_session_data(excelFile)
    user1Data = list()
    user2Data = list()
    for sheet in excel:
        user1Data.append(processUser1(excel[sheet]))
        user2Data.append(processUser2(excel[sheet]))
    return user1Data, user2Data
