import sys
# sys.path.append('..')
import re
import os
import json
from subprocess import Popen, PIPE, TimeoutExpired


def sh_execute(cmdstring, add_errors=True):
    proc = Popen(cmdstring.__str__(), stdout=PIPE, stderr=PIPE, shell=True)
    try:
        o, e = proc.communicate(timeout=15)
    except TimeoutExpired:
        proc.kill()
        o, e = proc.communicate()
    if add_errors:
        return (o.decode('utf-8') + e.decode('utf-8')).strip()
    else:
        return o.decode('utf-8').strip()


def add_index(fname, respect_file_extension=True, zerobased=False):

    if respect_file_extension and '.' in fname.split('/')[-1]:
        step = fname.split('.')
        dot_file_extension = '.' + step[-1]
        fname = '.'.join(step[:-1])
    else:
        dot_file_extension = ''
    if zerobased:
        fname_next = fname + '_0'
    else:
        fname_next = fname
    k = 1
    while os.path.exists(fname_next + dot_file_extension):
        fname_next = fname + '_' + str(k)
        k += 1
    return fname_next + dot_file_extension


def mk_new_dir(dirname, zerobased=False):
    newname = add_index(dirname, zerobased=zerobased)
    os.makedirs(newname, exist_ok=True)
    return newname

def checkflag(flags, fname, tail=False):
    """
    Checks if all flags are (or are not) present in a file fname. Flags can be of the following syntax:
    checkflag(['str1', '-str2', '\\-str3'], 'file.txt') checks, that:
    'str1' is in  file.txt
    AND
    'str2' is NOT in file.txt
    AND
    '-str3' is in file.txt (\\- is escape for literal minus, \\\\ for backslash )
    @param flags: string or list of strings, each representing a flag
    @param fname: string: file name
    @param tail: int or bool: number of last lines where the search is performed. If False, the whole file is searched
    @return: bool
    """
    flags = [flags] if isinstance(flags, str) else flags
    out = True
    for flag in flags:
        assert isinstance(flag, str)
        if tail:
            if tail is True:
                tail = 1
            cmd_string = 'tail -{tail} {fname} | grep -ie \'{flag}\''.format(tail=tail, flag='{}', fname=fname)
        else:
            cmd_string = 'grep -ie \'{flag}\' {fname}'.format(flag='{}', fname=fname)
        if flag[0] == '-':
            print('First character is \'-\', negation is applied. Use \'\\-\' if literal minus is intended')
            cmd_string = cmd_string.format(flag[1:])
            out = out and not bool(sh_execute(cmd_string, add_errors=False))
        elif flag[0] == '\\':
            print('First characters \'\\\\\' treated as an escape sentence')
            cmd_string = cmd_string.format(flag)
            out = out and bool(sh_execute(cmd_string, add_errors=False))
        else:
            cmd_string = cmd_string.format(flag)
            out = out and bool(sh_execute(cmd_string, add_errors=False))
    return out


def add_separator(lst, separator):
    if len(lst) > 1:
        lst[1:] = [separator + item for item in lst[1:]]
    return lst


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def file2dict(fname):
    with open(fname, 'r') as inf:
        return eval(inf.read())


def override_dict(master_dict, changeable):
    slave_dict = changeable.copy()
    for key, val in master_dict.items():
        if key in slave_dict.keys():
            if isinstance(val, dict) and isinstance(slave_dict[key], dict):
                override_dict(val, slave_dict[key])
            else:
                slave_dict[key] = val
    return slave_dict


def recursive_map_to_keys(f, dct: dict):
    """
    Recursive applies function f to all keys, subkeys, subsubkeys, ... of a dictionary
    @param f: fuction object (must return string)
    @param dct: dictionary
    @return:
    """
    if not isinstance(f('abc'), str):
        raise ValueError('first argument must be a function object which returns str')
    out_dct = dct.copy()
    for k, v in dct.items():
        newkey = f(k)
        del out_dct[k]
        if isinstance(v, dict):
            newval = recursive_map_to_keys(f, v)
        else:
            newval = v
        out_dct[newkey] = newval

    return out_dct


def recursively_map_to_vals(f, dct: 'dict'):
    """
    Recursive applies function f to all values of dictionary and subtictionaries
    @param f: fuction object (must return string)
    @param dct: dictionary
    @return:
    """
    # if not isinstance(dct, dict):
    #     return f(dct)

    out_dct = dct.copy()
    for k, v in dct.items():
        if isinstance(v, dict):
            newval = recursively_map_to_vals(f, v)
        else:
            newval = f(v)
        out_dct[k] = newval

    return out_dct


def try_numerize_string(string):
    if not isinstance(string, str):
        return string
    else:
        string = string.strip()
    if re.match('-?[0-9]+$', string):
        return int(string)
    else:
        try:
            return float(string)
        except:
            return string


def cjson_load(json_fname, comment=r'\\\#'):
    """
    Reading the commented json (cjson) file
    
    It is a regular json file except that \# - beginning comments are allowed
    @param json_fname: 
    @param comment: str comment str flag
    @return: dictionary corresponding to cjson file
    """
    if isinstance(json_fname, dict):
        return json_fname
    lst = []
    with open(json_fname, 'r') as jsn:
        for s in jsn:
            lst.append(re.sub(comment + r'.*', '', s))
        string = '\n'.join(lst).strip()
    dct = json.loads(string)
    return dct


if __name__ == '__main__':
    # jsn_comment = '/home/vsbat/SYNC/00_Current_PyWork/refine_with_gaussian/testfolder/test.json'
    # test_fname = '/home/vsbat/SYNC/00_Current_PyWork/refine_with_gaussian/testfolder/testcheckflag.txt'
    #
    # test_dct = cjson_load(jsn_comment)
    # chk = checkflag(test_dct['tests']['testkey1']['flags'], test_fname)
    # print(chk)
    print(os.getcwd())
    print(sh_execute('vasp5 4 >> out'))