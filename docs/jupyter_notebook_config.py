#saved in /home/<user>/.jupyter

import os
from subprocess import check_call

c = get_config()

def post_save(model, os_path, contents_manager):
        """post-save hook for converting notebooks to .py scripts"""
        if model['type'] != 'notebook':
                return # only do this for notebooks
        d, fname = os.path.split(os_path)
        check_call(['jupyter', 'nbconvert', fname, '--to', 'script', '--output-dir', '/Akamai/MLDance/backups/scripts'], cwd=d)

c.FileContentsManager.post_save_hook = post_save
# Configuration file for jupyter-notebook.

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## This is an application.

## The date format used by logging formatters for %(asctime)s
#c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#c.Application.log_level = 30

#------------------------------------------------------------------------------
# JupyterApp(Application) configuration
#------------------------------------------------------------------------------

## Base class for Jupyter applications

## Answer yes to any prompts.
#c.JupyterApp.answer_yes = False

## Full path of a config file.
#c.JupyterApp.config_file = ''

## Specify a config file to load.
#c.JupyterApp.config_file_name = ''
