import prepare_data as prep
import pandas as pd

'''
example merge_and _clean functions
'''


def duchenne_smiles_clean_r(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    """
    column_names = [" timestamp", " AU12_r", " AU6_r"]
    #    TODO: check timestamps align
    merged_df = pd.concat([df1[" timestamp"], df1[" AU12_r"], df1[" AU06_r"], df2[" AU12_r"], df2[" AU06_r"]], axis=1,
                          keys=["Timestamp", "User1Lipcorner", "User1Cheekraise", "User2Lipcorner", "User2Cheekraise"])
    return merged_df


def smiles_clean_r(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    """
    column_names = [" timestamp", " AU12_r"]
    #    TODO: check timestamps align
    merged_df = pd.concat([df1[" timestamp"], df1[" AU12_r"], df2[" AU12_r"]], axis=1,
                          keys=["Timestamp", "User1Smile", "User2Smile"])
    return merged_df


def eyeblink_clean_r(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    """
    column_names = [" timestamp", " AU45_r"]
    #    TODO: check timestamps align
    merged_df = pd.concat([df1[" timestamp"], df1[" AU45_r"], df2[" AU45_r"]], axis=1,
                          keys=["Timestamp", "User1Blink", "User2Blink"])
    return merged_df


#   creates a summary xls from all the data in folder "Intro"
prep.create_excel_study_summary("FF","Eyeblink_Raw", [" timestamp", " AU45_r"], eyeblink_clean_r)