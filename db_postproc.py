#!python3
import os
import warnings
from typing import Union, Callable, Any, List, Iterable
import pickle
import numpy as np
import sys
sys.path.append('src/ext_submodules')
from ase.io import read, write
from ase import Atoms
from pathlib import Path
from cclib.parser.utils import PeriodicTable, convertor
from ab_initio_postprocessing.abInitio_io_parse.gau_parse import get_gap, get_formula, get_kohn_sham
from genutils.misc import rhasattr, rgetattr, get_sorted_compositions
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from ab_initio_postprocessing.tools_stability.aux_routines import list_fmt2table
from ab_initio_postprocessing.graph_utils.my_graphs import draw_spectrum
from ab_initio_postprocessing.graph_utils.formatting import cm2inch, set_ax_position_cm
from ab_initio_postprocessing.abInitio_io_parse.gau_parse import getpdos_general
from genutils.filesystem_tools import sh_execute
import argparse
from datetime import datetime

tdy = datetime.today().strftime('%Y%m%d')

HARTREE_in_eV = 27.21138505

ptable = PeriodicTable()
element_labels = np.array(ptable.element[:])
HARTREE = convertor(1, 'hartree', 'eV')  # cclib-compatible

def process_db_path(db_path, *args, **kwargs):
    """
    @param db_path: string, a path of folder with pkl database files or of a single pkl file
    @param args: arguments to be passed to process_db_file_list
    @param kwargs: keyword arguments to be passed to process_db_file_list
    @return:
    """
    if os.path.isfile(db_path) and 'pkl' in str(db_path):
        pkl_file_list = [db_path]
    elif os.path.isdir(db_path):
        pkl_file_list = Path(db_path).glob('*.pkl')
    elif isinstance(db_path, list):
        pkl_file_list = db_path
    return process_db_file_list(pkl_file_list, *args, **kwargs)


def process_db_file_list(file_list,
                         element_nos=None, element_symbols=None,
                         res_folder=None,
                         n_isoms=1,
                         write_poscars = False,
                         write_xyz=False,
                         write_text = True,
                         first_connected_map=None,
                         sort_by: Union[str, Callable] = 'ccdata.scfenergies',
                         **kwargs):

    os.makedirs(res_folder, exist_ok=True)
    if element_nos and not element_symbols:
        element_symbols = tuple(element_labels[i] for i in element_nos)

    master_dict = db_file_list_to_dict(file_list,
                                       element_nos=element_nos,
                                       element_symbols=element_symbols, first_connected_map=first_connected_map,
                                       sort_by=sort_by)
    if write_poscars:
        write_db_to_poscars(master_dict,
                            str(res_folder) + '/POSCARS')
    if write_xyz:
        write_xyz_of_n_lowest(master_dict, n_lowest=n_isoms, outdir=str(res_folder) + '/xyz_files', **kwargs)
    if write_text:
        # best_list = sorted_bests(master_dict, first_connected_map=first_connected_map, lowest_inds_dict_out=False,
        #                          sortby=sortby)
        write_txt_data(master_dict, element_symbols=element_symbols, res_folder=str(res_folder), n_isoms=n_isoms, **kwargs)

    return master_dict


def get_task_db(file_list):
    task_db = []
    for pklfile in file_list:
        print('reading gau_db file ' + str(pklfile))
        with open(pklfile, 'rb') as pklfid:
            loaded_tasks = pickle.load(pklfid)
            for tsk in loaded_tasks:
                tsk.db_file = str(pklfile).split('/')[-1]
                tsk.where = '/'.join(str(pklfile).split('/')[:-1])
                tsk.fold_ind = int(tsk.name.split('_')[-1])
            task_db += loaded_tasks

    return task_db



def db_file_list_to_dict(file_list,
                         sort_by: Union[str, Callable] = 'ccdata.scfenergies', last_value = True,
                         element_nos=None, element_symbols=None,
                         res_folder=None,
                         first_connected_map=None,
                         skipnangap=False):

    """
    Utility analysing the folder containing database pkl files (recursively) and returning
    1. dictionary {(composition,): {'old_ind': [old_indices],
                                 'energies': [energies],
                                 'ccdata': [cclib_datas],
                                 'taskname': [task_names]}}  # for retreiving old indices



    @param db_pkl_fnames: list of paths to GauCalcDB's, or one path as a string
    @param element_nos: tuple with periodic numbers of elements in the desired order, e.g. (6, 1) for C H

    """
    if element_symbols and not element_nos:
        element_nos = tuple(ptable.number[sym] for sym in element_symbols)
    if res_folder is None:
        res_folder = '.'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    database = get_task_db(file_list)
    master_dict = {}
    for task in database:
        try:
            comp = tuple(np.count_nonzero(task.ccdata.atomnos == at_no) for at_no in element_nos)
        except AttributeError:
            print(task.name + ' in ' + task.db_file + ' : job failed')
            continue
        if hasattr(sort_by, '__call__'):
            sort_val = sort_by(task.ccdata)
        elif rhasattr(task, sort_by):
            sort_val = rgetattr(task, sort_by)
        else:
            warnings.warn("Task " + task.name + " has no sorting value: " + str(sort_by))
            continue
        if last_value:
            sort_val = sort_val[-1]
        if comp in master_dict:
            master_dict[comp]['tasks'].append(task)
            master_dict[comp]['sortvalues'].append(sort_val)
            master_dict[comp]['fold_ind'].append(int(task.name.split('_')[-1]))
        else:
            master_dict[comp] = {  'tasks': [task],
                                   'sortvalues': [sort_val],
                                   'fold_ind': [int(task.name.split('_')[-1])],
            #                      'energies': [task.ccdata.scfenergies[-1]],
            #                      'ccdata': [task.ccdata],
            #                      'taskname': [task.name],
            #                      'where': [task.db_file.split('/')[-1]],
                                 }
    for cmp, val in master_dict.items():
        idx = np.argsort(val['sortvalues'])
        if first_connected_map is not None:
            min_old_idx = first_connected_map[cmp]
        else:
            min_old_idx = 0
        if sort_by == 'ccdata.scfenergies' or sort_by == 'ccdata.freeenergy':
            val[sort_by] = [val['sortvalues'][int(i)] for i in idx if val['fold_ind'][int(i)] >= min_old_idx]
        val['tasks'] = [val['tasks'][int(i)] for i in idx if val['fold_ind'][int(i)] >= min_old_idx]
        val['fold_ind'] = [val['fold_ind'][int(i)] for i in idx if val['fold_ind'][int(i)] >= min_old_idx]
        del val['sortvalues']

    return master_dict

def write_db_to_poscars(master_dict, poscar_fname, n_lowest: int =1, vacuumsize=15., append=True):
    sorted_cmp = get_sorted_compositions(master_dict)
    for cmp in sorted_cmp:
        comp = tuple(cmp)
        val = master_dict[comp]
        n_avail = min(n_lowest, len(val['tasks']))
        for i in range(n_avail):
            ccd_i = val['tasks'][i].ccdata
            coords = ccd_i.atomcoords[-1]
            numbers = ccd_i.atomnos
            cell = np.diag(np.max(coords, axis=0) - np.min(coords, axis=0) + vacuumsize)
            pbc = np.ones(3, dtype=bool)
            atms = Atoms(positions=coords, numbers=numbers, pbc=pbc, cell=cell)
            atms.center()
            write(poscar_fname, atms, append=append, vasp5=True)

def write_xyz_of_n_lowest(master_dict, n_lowest, outdir='xyz_files', all_xyz_file=None, **kwargs):
    if all_xyz_file is not None:
        sh_execute(f'rm -rf {all_xyz_file.as_posix()}')

    os.makedirs(outdir, exist_ok=True)
    sorted_cmp = get_sorted_compositions(master_dict)
    for cmp in sorted_cmp:
        comp = tuple(cmp)
        val = master_dict[comp]
        n_avail = min(n_lowest, len(val['tasks']))
        for i in range(n_avail):
            xyz_fname = Path(outdir) / f"{get_formula(val['tasks'][i].ccdata)}_i{i}_{val['fold_ind'][i]}.xyz"
            val['tasks'][i].ccdata.metadata['comments'] = [f'{get_formula(val["tasks"][i].ccdata)} '
                                                           f'E(SCF) = {val["tasks"][i].ccdata.scfenergies[-1]:.4f} eV']
            val['tasks'][i].ccdata.writexyz(xyz_fname.as_posix())
            if all_xyz_file is not None:
                sh_execute(f"cat {xyz_fname.as_posix()} >> {all_xyz_file}")
                sh_execute(f"echo '' >> {all_xyz_file}")


def write_txt_data(master_dict, element_symbols, res_folder, attributes: Union[Iterable, str] =None, n_isoms=1, **kwargs):
    n_el = len(list(master_dict.keys())[0])
    if attributes is None:
        attributes = ['scfenergies', 'gap']
    if isinstance(attributes, str):
        attributes = [attributes]
    if 'gap' in attributes:
        for i in range(len(attributes)):
            if attributes[i] == 'gap':
                attributes[i] = get_gap
    # if n_el == 2:
    #     flatten_list = [x[0] + x[1:-1] for x in list_fmt_best]
    #     list_fmt2table(np.array(flatten_list)[:, [0, 1, 2]], outfile=res_folder + '/en_table.txt')
    #     list_fmt2table(np.array(flatten_list)[:, [0, 1, 3]], outfile=res_folder + '/gap_table.txt')
    # with open(res_folder + '/n_m_Enm_gap_2.txt', 'w') as stats_fid, open(res_folder + '/all_gaps.txt', 'w') as gaps_fid:
    #     stats_fid.write(('%s\t' * n_el) % element_symbols + ' ' + sortby + '\tGap, ev\tinitial ind\n')
    #     gaps_fid.write(('%s\t' * n_el) % element_symbols + '\tGap, ev\n')
    #     for dt in list_fmt_best:
    #         stats_fid.write(('%d\t' * n_el + '%.6f\t%.6f\t%s\t%d\n') % tuple(dt[0] + dt[1:]))
    #         gaps_fid.write(('%d\t' * n_el + '%.6f\n') % (tuple(dt[0]) + (dt[2],)))
    str_attr = []
    for attr in attributes:
        if hasattr(attr, '__call__'):
            str_attr.append(attr.__name__.replace('get_',''))
        else:
            str_attr.append(attr.split('.')[-1])
    sorted_comps = get_sorted_compositions(master_dict)

    header_list = list(element_symbols) + ['isomer'] + list(str_attr)
    cmp_isom_dict = {**{symbol: [] for symbol in element_symbols}, **{'isomer': []}}
    attr_values_dict = {j: [] for j in str_attr}
    for comp in sorted_comps:
        cmp = tuple(comp)
        for i in range(np.min((n_isoms, len(master_dict[cmp]['tasks'])))):
            for i_s, sym in enumerate(element_symbols):
                cmp_isom_dict[sym].append(cmp[i_s])
            cmp_isom_dict['isomer'].append(i)
            for i_attr, attr in enumerate(attributes):
                if hasattr(attr, '__call__'):
                    attr_values_dict[str_attr[i_attr]].append(attr(master_dict[cmp]['tasks'][i].ccdata))
                elif hasattr(master_dict[cmp]['tasks'][i].ccdata, attr):
                    try:
                        vals = rgetattr(master_dict[cmp]['tasks'][i].ccdata, attr)[-1]
                    except TypeError:
                        vals = rgetattr(master_dict[cmp]['tasks'][i].ccdata, attr)
                        if attr == 'freeenergy':
                            vals *= HARTREE_in_eV
                    attr_values_dict[str_attr[i_attr]].append(vals)
                else:
                    attr_values_dict[str_attr[i_attr]].append(rgetattr(master_dict[cmp]['tasks'][i], attr))
    table_dict = {**cmp_isom_dict, **attr_values_dict}

    stats_table = PrettyTable()

    for header in header_list:
        stats_table.add_column(header, table_dict[header])

    with open(res_folder + '/output.txt', 'w') as stats_fid:
                stats_table.vrules = 0
                stats_fid.write(str(stats_table))

    if n_el == 2:
        groundstate_dict = {k: [] for k in attr_values_dict.keys()}
        # Filter only lowest-energy structures
        gs_indices = np.where(np.array(table_dict['isomer']) == 0)[0]
        for i_gs in gs_indices:
            for k in groundstate_dict.keys():
                groundstate_dict[k].append(table_dict[k][i_gs])
        cmps_array = np.array(sorted_comps)
        for i, attr in enumerate(str_attr):
            try:
                values_column = np.array(groundstate_dict[attr], dtype=float)
            except:
                continue
            attr_listfmt = np.hstack((cmps_array, values_column.reshape((-1,1))))
            list_fmt2table(attr_listfmt, outfile=res_folder + f'/{tdy}_{attr}_TABLE.txt')




def plot_el_spectra_binary(master_dict, element_symbols, savefiles=True, save_folder='spectra_graphs', groupped=True,
                           format='png', norm_by_natom=False, pdoskeys=None, **drawspectrum_kwargs):
    """
    
    :param master_dict: 
    :param element_symbols: 
    :param savefiles: 
    :param save_folder: 
    :param groupped: 
    :param format: 
    :param norm_by_natom: 
    :param pdoskeys: keywords to plot pdos, format "[atom_no, 1-based][atom_symbol][,orbital]" e.g. "4Au,D" 
    :param drawspectrum_kwargs: 
    :return: 
    """
    # plt.rcParams['xtick.major.pad'] = '0'
    # plt.rcParams['ytick.major.pad'] = '0.'
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7

    if savefiles and not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # master_dict =  db_files_to_dict(db_pkl_fnames, element_symbols=element_symbols)
    composition_array = get_sorted_compositions(master_dict)

    figno = 0
    for first_el, cnt in np.column_stack(np.unique(composition_array[:, 0], return_counts=True)):
        if groupped:
            figno += 1
            plt.figure()
            fig, axs = plt.subplots(cnt, 1, sharex=True)
            plt.setp(axs, yticks=np.array([-0.5, 0, 0.5]))
            fig.subplots_adjust(hspace=0.1)
            fig.set_size_inches(cm2inch(8, 1.5*cnt))
        for i, composition in enumerate(composition_array[composition_array[:,0] == first_el]):
            db_key = tuple(composition)
            db_val = master_dict[db_key]
            lowest_isom_idx = np.argsort(db_val['ccdata.scfenergies'])[0]
            cc_data = db_val['tasks'][lowest_isom_idx].ccdata
            up_dos, dn_dos, fermi = get_kohn_sham(cc_data)
            if pdoskeys is not None:
                pdos_up, _ = getpdos_general(cc_data)
            up_dos -= fermi
            if cc_data.nelectrons % 2:
                dn_dos -= fermi
            else:
                dn_dos = None

            if groupped:
                base = axs[i]
                base.tick_params(direction='in')
            else:
                fig = plt.figure()
                fig.set_size_inches(cm2inch(8, 3))
                base = plt

            normalization = np.sum(composition) if norm_by_natom else True

            draw_spectrum(specup=up_dos, specdn=dn_dos, e_fermi=0, shareax=base,
                          label=get_formula(cc_data), sigma=0.05, normalization=normalization, **drawspectrum_kwargs)
            plt.xticks(np.arange(*drawspectrum_kwargs['span'], 1))
            if not groupped:
                ax = plt.gca()
                set_ax_position_cm(ax, [0.8, 0.8, 7, 2])
                ax.tick_params(bottom=True, top=True, direction='in')
                plt.savefig(save_folder + '/' + get_formula(cc_data) + '.' + format, dpi=600)
                plt.close()

        if groupped:
            plt.savefig(save_folder + '/' + element_symbols[0] + str(first_el) + '.' + format, dpi=600)
            plt.close()


    # compositions_dict = dict(zip(element_symbols, [[]] * len(element_symbols)))
    # for key, val in master_dict:
    #     fla = get_formula(val[0]['ccdata'], extended_out=True)['fdict']
    #     for el_sym in element_symbols:
    #         compositions_dict[el_sym].append(fla[el_sym])



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", required=True, help="Directory containing results")
    ap.add_argument("-l", "--labels", required=True, help="Directory containing results")
    ap.add_argument("-p", "--properties", required=False, help="Directory containing results", default=None)
    input_kwargs = vars(ap.parse_args())
    search_dir = input_kwargs['folder']
    atom_labels = tuple(input_kwargs['labels'].strip().split())
    pkl_files = Path(search_dir).rglob('*.pkl')
    process_db_file_list(pkl_files, element_symbols=atom_labels, res_folder=search_dir + '/postproc', write_xyz=True,
                         write_text=True, n_isoms=1, attributes=input_kwargs['properties'])
