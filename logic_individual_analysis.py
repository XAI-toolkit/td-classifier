# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:00:54 2021

@author: tsoukj
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:10 2021

@author: tsoukj
"""

import os
import time
import subprocess
import pandas as pd
import numpy as np
from pydriller.metrics.process.commits_count import CommitsCount
from pydriller.metrics.process.code_churn import CodeChurn
from pydriller.metrics.process.contributors_count import ContributorsCount
from pydriller.metrics.process.contributors_experience import ContributorsExperience
from pydriller.metrics.process.hunks_count import HunksCount

def run_pydriller2(cwd, git_url, repo_name, from_commit, to_commit):    
    print('~~~~~~ Running PyDriller ~~~~~~')
    
    # Create PyDriller results directory
    pydriller_dir = r'%s\tool_results\pydriller\%s' % (cwd, repo_name)
    if not os.path.exists(pydriller_dir):
        os.makedirs(pydriller_dir)

    # Compute Commits Count
    start_time = time.time()
    metric = CommitsCount(path_to_repo=git_url, from_commit=from_commit, to_commit=to_commit)
    temp_dataset = pd.DataFrame.from_dict(metric.count(), orient='index', columns=['commits_count'])
    print('- Successfully fetched Commits Count metric (%s sec)' % (round(time.time() - start_time, 2)))

    # Compute Code Churn
    start_time = time.time()
    metric = CodeChurn(path_to_repo=git_url, from_commit=from_commit, to_commit=to_commit)
    df_files_count = pd.DataFrame.from_dict(metric.count(), orient='index', columns=['code_churn_count'])
    temp_dataset = temp_dataset.join(df_files_count)
    df_files_max = pd.DataFrame.from_dict(metric.max(), orient='index', columns=['code_churn_max'])
    temp_dataset = temp_dataset.join(df_files_max)
    df_files_avg = pd.DataFrame.from_dict(metric.avg(), orient='index', columns=['code_churn_avg'])
    temp_dataset = temp_dataset.join(df_files_avg)
    print('- Successfully fetched Code Churn metric (%s sec)' % (round(time.time() - start_time, 2)))
    
    # Compute Contributors Count
    start_time = time.time()
    metric = ContributorsCount(path_to_repo=git_url, from_commit=from_commit, to_commit=to_commit)
    df_contributors_count = pd.DataFrame.from_dict(metric.count(), orient='index', columns=['contributors_count'])
    temp_dataset = temp_dataset.join(df_contributors_count)
    print('- Successfully fetched Contributors Count metric (%s sec)' % (round(time.time() - start_time, 2)))
    
    # Compute Contributors Experience
    start_time = time.time()
    metric = ContributorsExperience(path_to_repo=git_url, from_commit=from_commit, to_commit=to_commit)
    df_contributors_experience = pd.DataFrame.from_dict(metric.count(), orient='index', columns=['contributors_experience'])
    temp_dataset = temp_dataset.join(df_contributors_experience)
    print('- Successfully fetched Contributors Experience metric (%s sec)' % (round(time.time() - start_time, 2)))
    
    # Compute Hunks Count
    start_time = time.time()
    metric = HunksCount(path_to_repo=git_url, from_commit=from_commit, to_commit=to_commit)
    df_hunks_count = pd.DataFrame.from_dict(metric.count(), orient='index', columns=['hunks_count'])
    temp_dataset = temp_dataset.join(df_hunks_count)
    print('- Successfully fetched Hunks Count metric (%s sec)' % (round(time.time() - start_time, 2)))
    
    # Reset index and create column 'class_path'
    temp_dataset = temp_dataset.reset_index()
    temp_dataset.rename(columns={ temp_dataset.columns[0]: 'class_path' }, inplace = True)
    
    # Remove NAN rows
    temp_dataset.dropna(subset=['class_path'], inplace=True)
    # Keep only .java files
    temp_dataset = temp_dataset[temp_dataset['class_path'].str.contains('.java', regex=False)]
    # Replace '\' with '/'
    temp_dataset['class_path'] = temp_dataset['class_path'].str.replace('\\', '/')
    # Create column 'class_name'
    class_col = temp_dataset.apply(lambda row: row['class_path'].split('/')[-1], axis=1)
    temp_dataset.insert(1, 'class_name', class_col)
    
    # Export dataframe to csv
    temp_dataset.to_csv(r'%s\%s_pydriller_measures.csv' % (pydriller_dir, repo_name), sep=',', na_rep='', index=False)

def run_ck2(cwd, clone_dir, repo_name):
    print('~~~~~~ Running CK ~~~~~~')
    
    # Create ck results directory
    ck_dir = r'%s\tool_results\ck\%s' % (cwd, repo_name)
    if not os.path.exists(ck_dir):
        os.makedirs(ck_dir)
    os.chdir(ck_dir)
    
    # Execute ck jar
    start_time = time.time()
    COMMAND1 = r'java -jar %s\lib\ck-0.6.4-SNAPSHOT-jar-with-dependencies.jar %s true 0 false' % (cwd, clone_dir)
    p = subprocess.Popen(COMMAND1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND1, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CK metrics (%s sec)' % round(time.time() - start_time, 2))
        
    os.chdir(cwd)

def run_refactoring_miner2(cwd, clone_dir, repo_name, from_commit, to_commit):
    print('~~~~~~ Running RefactoringMiner ~~~~~~')
    
    # Create rm results directory
    rm_dir = r'%s\tool_results\rm\%s' % (cwd, repo_name)
    if not os.path.exists(rm_dir):
        os.makedirs(rm_dir)
    
    # Execute rm tool
    start_time = time.time()
    COMMAND2 = r'%s\lib\RefactoringMiner\bin\RefactoringMiner -bc %s %s %s > %s\%s_refactoring_measures.json' % (cwd, clone_dir, from_commit, to_commit, rm_dir, repo_name)
    p = subprocess.Popen(COMMAND2, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND2, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched RefactoringMiner metrics (%s sec)' % round(time.time() - start_time, 2))
    
def run_cpd2(cwd, clone_dir, repo_name):
    print('~~~~~~ Running CPD ~~~~~~')
    
    # Create cpd results directory
    cpd_dir = r'%s\tool_results\cpd\%s' % (cwd, repo_name)
    if not os.path.exists(cpd_dir):
        os.makedirs(cpd_dir)
    
    # Execute cpd tool
    start_time = time.time()
    COMMAND3 = r'%s\lib\pmd\bin\cpd.bat --minimum-tokens 100 --files %s --skip-lexical-errors --format csv > %s\%s_duplication_measures.csv' % (cwd, clone_dir, cpd_dir, repo_name)
    p = subprocess.Popen(COMMAND3, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND3, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CPD metrics (%s sec)' % round(time.time() - start_time, 2))

def run_cloc2(cwd, clone_dir, repo_name):
    print('~~~~~~ Running cloc ~~~~~~')
    
    # Create cpd results directory
    cloc_dir = r'%s\tool_results\cloc\%s' % (cwd, repo_name)
    if not os.path.exists(cloc_dir):
        os.makedirs(cloc_dir)
    
    # Execute cloc tool
    start_time = time.time()
    COMMAND4 = r'%s\lib\cloc-1.88.exe %s --by-file --force-lang="Java",java --include-ext=java --csv --out="%s\%s_comments_measures.csv"' % (cwd, clone_dir, cloc_dir, repo_name)
    p = subprocess.Popen(COMMAND4, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    exit_code = p.returncode
    stdout = stdout
    stderr = stderr
    
    if exit_code != 0:
        print('- Error executing command [%s]\n- Exit code [%s]\n- stderr: [%s]\n- stdout: [%s]' % (COMMAND4, exit_code, stderr, stdout))
    else:
        print('- Successfully fetched CLOC metrics (%s sec)' % round(time.time() - start_time, 2))
    
def merge_results(cwd, repo_name):

    # metrics_df = pd.DataFrame()
    
    # Read and merge pydriller csv results
    pydriller_dir = r'%s\tool_results\pydriller\%s' % (cwd, repo_name)
    metrics_df = pd.read_csv(r'%s\%s_pydriller_measures.csv' % (pydriller_dir, repo_name), sep=',')
    
    print('- Successfully merged pydriller metrics')

    # Read ck csv results
    ck_dir = r'%s\tool_results\ck\%s' % (cwd, repo_name)
    temp_dataset = pd.read_csv(r'%s\class.csv' % ck_dir, sep=",")
    # Remove useless prefix from class path and class name
    for index, row in temp_dataset.iterrows(): 
        temp_dataset.loc[index,'file'] = row['file'].split('%s\\' % repo_name, 1)[-1]
        temp_dataset.loc[index,'class'] = '%s.java' % row['class'].split('.')[-1]
    # Replace '\' with '/'
    temp_dataset['file'] = temp_dataset['file'].str.replace('\\', '/')
    
    # Merge ck metrics with metrics_df
    for index, row in temp_dataset.iterrows():
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'cbo'] = row['cbo']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'wmc'] = row['wmc']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'dit'] = row['dit']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'rfc'] = row['rfc']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'lcom'] = row['lcom']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'total_methods'] = row['totalMethodsQty']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'max_nested_blocks'] = row['maxNestedBlocksQty']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'loc'] = row['loc']
        metrics_df.loc[(metrics_df['class_path'] == row['file']) & (metrics_df['class_name'] == row['class']), 'total_variables'] = row['variablesQty']
    # Remove NAN rows
    metrics_df.dropna(inplace=True)
    
    print('- Successfully merged CK metrics')
    

    # # Read RefactoringMiner json results
    # rm_dir = r'%s\tool_results\rm\%s' % (cwd, repo_name)
    # temp_dataset = pd.DataFrame()
    # # Read and flatten json files
    # df_from_json = pd.DataFrame.from_records(pd.read_json(r'%s\%s_refactoring_measures.json' % (rm_dir, repo_name))['commits'])
    # if not df_from_json.empty:
    #     for index, row in df_from_json.iterrows():
    #         if row['refactorings']:
    #             for refactoring in row['refactorings']:
    #                 for leftSideLocation in refactoring['leftSideLocations']:
    #                     new_row = {'class_path':leftSideLocation['filePath'], 'type':refactoring['type']}
    #                     temp_dataset = temp_dataset.append(new_row, ignore_index=True)
    #                 for rightSideLocation in refactoring['rightSideLocations']:
    #                     new_row = {'class_path':rightSideLocation['filePath'], 'type':refactoring['type']}
    #                     temp_dataset = temp_dataset.append(new_row, ignore_index=True)
    #     # Group refactorings based on class name and compute total refactorings per class
    #     temp_dataset = temp_dataset.groupby(['class_path', 'type']).size().unstack(fill_value=0)
    #     temp_dataset['total_refactorings'] = temp_dataset.sum(axis=1)
    #     temp_dataset.reset_index(inplace=True)
        
    #     # Merge RefactoringMiner metrics dataframe with metrics_df    
    #     for index, row in temp_dataset.iterrows():
    #         metrics_df.loc[metrics_df['class_path'] == row['class_path'], 'total_refactorings'] = row['total_refactorings']
            
    #     # Fill NaN values of total_refactorings with zeros
    #     metrics_df['total_refactorings'].fillna(0, inplace=True)
    # else:
    #     metrics_df['total_refactorings'] = 0
        
    # print('- Successfully merged RefactoringMiner metrics')


    # Read CPD csv results 
    cpd_dir = r'%s\tool_results\cpd\%s' % (cwd, repo_name)
    temp_dataset = pd.read_table(r'%s\%s_duplication_measures.csv' % (cpd_dir, repo_name))
    if not temp_dataset.empty:
        temp_dataset = temp_dataset.iloc[:,0].str.split(',', expand=True)
        temp_dataset_2 = pd.DataFrame()
        # create tupples with duplicated line intervals
        for index, row in temp_dataset.iterrows():
            dup_lines = row[0]
            dup_classes = row[3:]
            for i, v in dup_classes.items():
                # start from odd columns that are not None
                if i % 2 == 1 and dup_classes[i] != None:
                    temp_dataset_2 = temp_dataset_2.append({'class_path': dup_classes[i+1], 'line_tuple': [int(dup_classes[i]), int(dup_classes[i])+int(dup_lines)]}, ignore_index=True)
        # group dataframe by class and merge each classe line tuples into a list         
        temp_dataset_2 = pd.DataFrame(temp_dataset_2.groupby('class_path').apply(lambda x: list(np.unique(x)))).reset_index()
        # merge overlapping tuple intervals
        for i, temp_tuple in temp_dataset_2[0].items():
            temp_tuple.sort(key=lambda interval: interval[0])
            merged = [temp_tuple[0]]
            for current in temp_tuple:
                previous = merged[-1]
                if current[0] <= previous[1]:
                    previous[1] = max(previous[1], current[1])
                else:
                    merged.append(current)
            temp_dataset_2[0][i] = merged
        # calculate total duplicate lines per class from tuple intervals
        sum_tuple_list = []
        for i, temp_tuple in temp_dataset_2[0].items():
            sum_tuple = 0
            for tuple_list in temp_tuple:
                sum_tuple = sum_tuple + (tuple_list[1] - tuple_list[0] + 1)
            sum_tuple_list.append(sum_tuple)
        temp_dataset_2[0] = sum_tuple_list
        temp_dataset_2.rename(columns = {0:'duplicated_lines'}, inplace=True)
        # Replace '\' with '/'
        temp_dataset_2['class_path'] = temp_dataset_2['class_path'].str.replace('\\', '/')
        # Removed useless prefix from class path
        for index, row in temp_dataset_2.iterrows(): 
            temp_dataset_2.loc[index,'class_path'] = row['class_path'].split('%s/' % repo_name, 1)[-1]
            
        # Merge CPD metrics dataframe with metrics_df
        for index, row in temp_dataset_2.iterrows():
            metrics_df.loc[metrics_df['class_path'] == row['class_path'], 'duplicated_lines'] = row['duplicated_lines']
    
        # Fill NaN values of duplicated_lines with zeros
        metrics_df['duplicated_lines'].fillna(0, inplace=True)
    else:
        metrics_df['duplicated_lines'] = 0
    
    print('- Successfully merged CPD metrics')


    # Read cloc csv results
    cloc_dir = r'%s\tool_results\cloc\%s' % (cwd, repo_name)
    temp_dataset = pd.read_csv(r'%s\%s_comments_measures.csv' % (cloc_dir, repo_name), sep=",")
    # Remove last row with redundant data
    temp_dataset = temp_dataset[:-1]
    # Remove useless prefix from class path and class name
    for index, row in temp_dataset.iterrows():
        temp_dataset.loc[index,'filename'] = row['filename'].split('%s\\' % repo_name, 1)[-1]
    # Replace '\' with '/'
    temp_dataset['filename'] = temp_dataset['filename'].str.replace('\\', '/')
    
    # Merge cloc metrics dataframe with the updated dataset    
    for index, row in temp_dataset.iterrows():
        metrics_df.loc[metrics_df['class_path'].str.lower() == row['filename'], 'comment_lines'] = row['comment']
        metrics_df.loc[metrics_df['class_path'].str.lower() == row['filename'], 'ncloc'] = row['code']
        metrics_df.loc[metrics_df['class_path'].str.lower() == row['filename'], 'total_lines'] = row['blank'] + row['code'] + row['comment']
    
    # Fill NaN values of duplicated_lines with zeros
    metrics_df['comment_lines'].fillna(0, inplace=True)
    
    print('- Successfully merged CLOC metrics')
    
    # Drop NAN values
    metrics_df.dropna(inplace=True)
    metrics_df.reset_index(drop=True, inplace=True)
    
    return metrics_df