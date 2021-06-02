# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:44:54 2021

@author: tsoukj
"""

import argparse
import sys
import os
import time
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve
from logic_complete_analysis import clone_project, run_pydriller, run_ck, run_refactoring_miner, run_cpd, run_cloc, run_classifier, export_results
from logic_individual_analysis import run_pydriller2, run_ck2, run_refactoring_miner2, run_cpd2, run_cloc2, merge_results

# Create the Flask app
app = Flask(__name__)
# Enable CORS
CORS(app)

#===============================================================================
# td_classifier ()
#===============================================================================
@app.route('/TDClassifier/CompleteAnalysis', methods=['GET'])
def td_classifier(git_url_param=None, analysis_type_param=None):
    """
    API Call to TDClassifier service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
        analysis_type_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the classification results, intermediate static analysis
        results, a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None
    analysis_type_param = request.args.get('analysis_type', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None or analysis_type_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working directory
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        start_time = time.time()
        
        # Create empty metrics_df dataframe
        metrics_df = pd.DataFrame()
        
        # Run GitPython
        last_commit, first_commit = clone_project(git_url_param, clone_dir, repo_name, analysis_type_param)
        # Run PyDriller
        metrics_df = run_pydriller(cwd, git_url_param, repo_name, first_commit, last_commit, metrics_df)
        # Run CK
        metrics_df = run_ck(cwd, clone_dir, repo_name, metrics_df)
        # Run Refactoring Miner
        # metrics_df = run_refactoring_miner(cwd, clone_dir, repo_name, first_commit, last_commit, metrics_df)
        # Run CPD
        metrics_df = run_cpd(cwd, clone_dir, repo_name, metrics_df)
        # Run CLOC
        metrics_df = run_cloc(cwd, clone_dir, repo_name, metrics_df)
                
        # Drop NAN values
        metrics_df.dropna(inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        
        # Run classifier
        y_pred, y_pred_proba = run_classifier(cwd, metrics_df)
        metrics_df['high_td'] = y_pred
        metrics_df['high_td_proba'] = y_pred_proba
        
        # Export results in csv, json and html
        export_results(cwd, repo_name, metrics_df)
        
        print('- Successfully finished process in %s sec. Found %s high-TD classes (out of %s)' % (round(time.time() - start_time, 2), metrics_df['high_td'].sum(), metrics_df.shape[0]))
        
        # Convert dataframe to dict
        results = metrics_df.to_dict(orient='records')

        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# clone_project_service ()
#===============================================================================
@app.route('/TDClassifier/CloneProject', methods=['GET'])
def clone_project_service(git_url_param=None, analysis_type_param=None):
    """
    API Call to CloneProject service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
        analysis_type_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the first and last commit sha of the chosen analysis
        range, a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None
    analysis_type_param = request.args.get('analysis_type', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None or analysis_type_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working directory
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        # Run GitPython
        last_commit, first_commit = clone_project(git_url_param, clone_dir, repo_name, analysis_type_param)
        
        results = {
                'last_commit': last_commit,
                'first_commit': first_commit
                }
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_pydriller_service ()
#===============================================================================
@app.route('/TDClassifier/RunPydriller', methods=['GET'])
def run_pydriller_service(git_url_param=None, first_commit_param=None, last_commit_param=None):
    """
    API Call to RunPydriller service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
        first_commit_param: Required (sent as URL query parameter from API Call)
        last_commit_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None
    first_commit_param = request.args.get('first_commit', type=str) # Required: if key doesn't exist, returns None
    last_commit_param = request.args.get('last_commit', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None or first_commit_param is None or last_commit_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set current working directory
        cwd = os.environ.get('CWD')
        
        # Run PyDriller
        run_pydriller2(cwd, git_url_param, repo_name, first_commit_param, last_commit_param)
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            # 'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_ck_service ()
#===============================================================================
@app.route('/TDClassifier/RunCK', methods=['GET'])
def run_ck_service(git_url_param=None):
    """
    API Call to RunCK service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working directory
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        # Run CK
        run_ck2(cwd, clone_dir, repo_name)
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            # 'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_refactoring_miner_service ()
#===============================================================================
@app.route('/TDClassifier/RunRefactoringMiner', methods=['GET'])
def run_refactoring_miner_service(git_url_param=None, first_commit_param=None, last_commit_param=None):
    """
    API Call to RunRefactoringMiner service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
        first_commit_param: Required (sent as URL query parameter from API Call)
        last_commit_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None
    first_commit_param = request.args.get('first_commit', type=str) # Required: if key doesn't exist, returns None
    last_commit_param = request.args.get('last_commit', type=str) # Required: if key doesn't exist, returns None
    
    # If required parameters are missing from URL
    if git_url_param is None or first_commit_param is None or last_commit_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working directory
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        # Run CK
        run_refactoring_miner2(cwd, clone_dir, repo_name, first_commit_param, last_commit_param)
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            # 'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_cpd_service ()
#===============================================================================
@app.route('/TDClassifier/RunCPD', methods=['GET'])
def run_cpd_service(git_url_param=None):
    """
    API Call to RunCPD service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working 
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        # Run CPD
        run_cpd2(cwd, clone_dir, repo_name)
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            # 'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_cloc_service ()
#===============================================================================
@app.route('/TDClassifier/RunCLOC', methods=['GET'])
def run_cloc_service(git_url_param=None):
    """
    API Call to RunCLOC service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set clone and current working directory
        cwd = os.environ.get('CWD')
        clone_dir = r'%s\cloned\%s' % (cwd, repo_name)
        
        # Run CPD
        run_cloc2(cwd, clone_dir, repo_name)
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            # 'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# run_classifier_service ()
#===============================================================================
@app.route('/TDClassifier/RunClassifier', methods=['GET'])
def run_classifier_service(git_url_param=None):
    """
    API Call to RunClassifier service
    Arguments:
        git_url_param: Required (sent as URL query parameter from API Call)
    Returns:
        A JSON containing the classification results, intermediate static analysis
        results, a status code and a message.
    """

    # Parse URL-encoded parameters
    git_url_param = request.args.get('git_url', type=str) # Required: if key doesn't exist, returns None

    # If required parameters are missing from URL
    if git_url_param is None:
        return unprocessable_entity()
    else:
        # Set repo name from repo url
        if '.git' not in git_url_param:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]))).lower().strip()
        else:
            repo_name = (('%s_%s') % (git_url_param.split('/')[-2], (git_url_param.split('/')[-1]).split('.')[-2])).lower().strip()
        
        # Set current working directory
        cwd = os.environ.get('CWD')
        
        # Merge results
        metrics_df = merge_results(cwd, repo_name)
        
        # Run classifier
        y_pred, y_pred_proba = run_classifier(cwd, metrics_df)
        metrics_df['high_td'] = y_pred
        metrics_df['high_td_proba'] = y_pred_proba
                
        # Export results in csv, json and html
        export_results(cwd, repo_name, metrics_df)
                
        # Convert dataframe to dict
        results = metrics_df.to_dict(orient='records')
        
        # Compose and jsonify respond
        message = {
            'status': 200,
            'message': 'The request was fulfilled.',
            'results': results,
    	}
        resp = jsonify(message)
        resp.status_code = 200

        return resp

#===============================================================================
# errorhandler ()
#===============================================================================
@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + ' --> Please check your data payload.',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp
@app.errorhandler(422)
def unprocessable_entity(error=None):
    message = {
        'status': 400,
        'message': 'Unprocessable Entity: ' + request.url + ' --> Missing or invalid parameters.',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp
@app.errorhandler(500)
def internal_server_error(error=None):
    message = {
        'status': 500,
        'message': 'The server encountered an internal error and was unable to complete your request. ' + error,
    }
    resp = jsonify(message)
    resp.status_code = 500

    return resp

#===============================================================================
# run_server ()
#===============================================================================
def run_server(host, port, mode, debug_mode, cwd):
    """
    Executes the command to start the server
    Arguments:
        host: retrieved from create_arg_parser() as a string
        port: retrieved from create_arg_parser() as a int
        mode: retrieved from create_arg_parser() as a string
        debug_mode: retrieved from create_arg_parser() as a bool
    """

    print('server:      %s:%s' % (host, port))
    print('mode:        %s' % (mode))
    print('debug_mode:  %s' % (debug_mode))
    print('working_directory:  %s' % (cwd))
    if debug_mode:
        print(" *** Debug enabled! ***")

    # Store settings in environment variables
    os.environ['DEBUG'] = str(debug_mode)
    os.environ['CWD'] = str(cwd)

    if mode == 'builtin':
        # Run app in debug mode using flask
        app.run(host, port, debug_mode)
    elif mode == 'waitress':
        # Run app in production mode using waitress
        serve(app, host=host, port=port)
    else:
        print('Server mode "%s" not yet implemented' % mode)
        sys.exit(1)

#===============================================================================
# create_arg_parser ()
#===============================================================================
def create_arg_parser():
    """
    Creates the parser to retrieve arguments from the command line
    Returns:
        A Parser object
    """
    server_modes = ['builtin', 'waitress']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('h', metavar='HOST', help='Server HOST (e.g. "localhost")', type=str)
    parser.add_argument('p', metavar='PORT', help='Server PORT (e.g. "5000")', type=int)
    parser.add_argument('m', metavar='SERVER_MODE', help=", ".join(server_modes), choices=server_modes, type=str)
    parser.add_argument('--debug', help="Run builtin server in debug mode", action='store_true', default=False)

    return parser

#===============================================================================
# main ()
#===============================================================================
def main():
    """
    The main() function of the script acting as the entry point
    """
    parser = create_arg_parser()

    # If script run without arguments, print syntax
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()
    host = args.h
    mode = args.m
    port = args.p
    debug_mode = args.debug
    
    # Set current working directory
    cwd = os.getcwd()

    # Run server with user-given arguments
    run_server(host, port, mode, debug_mode, cwd)

if __name__ == '__main__':
    main()
