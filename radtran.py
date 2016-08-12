__author__ = 'nsteiner'

import os
import copy
import subprocess as sp
import itertools as it
import pickle


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LIBRADTRAN_PATH = '/home/nsteiner/libradtran'
UVSPEC_PATH = os.path.join(LIBRADTRAN_PATH, './bin/uvspec')

HEADER = ('lambda', 'edir', 'edn', 'eup', )

#HEADER = ['lambda', 'edir', 'edn', 'eup', 'uavgdir', 'uavgdn', 'uavgup']

"""
libRadtran Input
~~~~~~~~~~~~~~~~
"""


LOWTRAN_SOLAR = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'albedo': (0.88,),
    'sza': (0.0,),
    'rte_solver': ('disort',),
    'wavelength': (3000.0, 5000.0),
    'mol_abs_param': ('LOWTRAN',)}


TEST_SOLAR = {
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'output_user': HEADER,
    'rte_solver': ('twostr',),
    'wavelength': (200, 5000.0),
    'mol_abs_param': ('LOWTRAN',),
    'albedo': (1,)}

SOLAR_USS_TOA = {
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'albedo': (0,),
    'output_user': HEADER,
    'rte_solver': ('disort',),
    'wavelength': (2500, 6000.0),
    'mol_abs_param': ('LOWTRAN',),
    'zout': ('toa',)}


SOLAR_USS = {
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'albedo': (0.9,),
    'output_user': HEADER,
    'rte_solver': ('disort',),
    'wavelength': (2500, 6000.0),
    'mol_abs_param': ('LOWTRAN',)}

THERMAL_USS_TOA = {
    'source': ('thermal', ),
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'umu': (1.0,),
    'albedo': (0.0,),
    'rte_solver': ('disort',),
    'wavelength': (2500, 6000.0),
    'mol_abs_param': ('LOWTRAN',)}


THERMAL_USS_PROFILE = {
    'source': ('thermal', ),
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'albedo': (0,),
    'output_user': HEADER,
    'rte_solver': ('disort',),
    'wavelength': (2500, 6000.0),
    'mol_abs_param': ('LOWTRAN',),
    'zout': (0, 10, 20, 50)}

THERMAL_USS = {
    'source': ('thermal', ),
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'albedo': (0,),
    'output_user': HEADER,
    'rte_solver': ('disort',),
    'wavelength': (2500, 6000.0),
    'mol_abs_param': ('LOWTRAN',)}


FLIR_SOLAR_TOA = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'latitude': ('N', 66),
    'longitude': ('W', 140),
    'time': (2014, 07, 06, 14, 00, 00),
    'albedo': (0.88,),
    'output_user': HEADER,
    'rte_solver': ('sdisort',),
    'wavelength': (2000.0, 6000.0),
    'mol_abs_param': ('LOWTRAN',),
    'zout': ('toa',)}


FLIR_THERMAL_TOA = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('thermal', os.path.join(LIBRADTRAN_PATH, 'examples/UVSPEC_LOWTRAN_THERMAL.TRANS')),
    'latitude': ('N', 66),
    'longitude': ('W', 140),
    'time': (2014, 07, 06, 14, 00, 00),
    'albedo': (0,),
    'umu': (1,),  # sensor at nadir
    'sur_temperature': (273.15, ),
    'rte_solver': ('disort',),  # distort
    'wavelength': (2000.0, 6000.0),
    'mol_abs_param': ('LOWTRAN',),
    'zout': ('toa',)}


FLIR_SOLAR_PROFILE = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'latitude': ('N', 66),
    'longitude': ('W', 140),
    'time': (2014, 07, 06, 14, 00, 00),
    'albedo': (0.88,),
    'rte_solver': ('sdisort',),
    'wavelength': (3000.0, 5000.0),
    'mol_abs_param': ('LOWTRAN',),
    'zout': (0, 10, 20, 50)}


FLIR_THERMAL = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('thermal', os.path.join(LIBRADTRAN_PATH, 'examples/UVSPEC_LOWTRAN_THERMAL.TRANS')),
    'latitude': ('N', 66),
    'longitude': ('W', 140),
    'time': (2014, 07, 06, 14, 00, 00),
    'albedo': (.9, ),
    'output_user': ('lambda', 'edir', 'edn', 'eup', ),
    'rte_solver': ('disort',),
    'wavelength': (2000.0, 6000.0),
    'mol_abs_param': ('LOWTRAN',),
    'umu': (1,),
    'zout': (.2,)}


FLIR_SOLAR = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'data/atmmod/afglus.dat'),),
    'source': ('solar', os.path.join(LIBRADTRAN_PATH, 'data/solar_flux/kurudz_1.0nm.dat')),
    'latitude': ('N', 66),
    'longitude': ('W', 140),
    'time': (2014, 07, 06, 14, 00, 00),
    'albedo': (.9, ),
    'output_user': ('lambda', 'edir', 'edn', 'eup', 'uu'),
    'rte_solver': ('disort',),
    'wavelength': (2000.0, 6000.0),
    'umu': (1,),
    'zout': (.2,),
    'mol_abs_param': ('LOWTRAN',)}


LOWTRAN_THERMAL = {
    'atmosphere_file': (os.path.join(LIBRADTRAN_PATH, 'examples/AFGLUS.70KM'),),
    'source': ('thermal', os.path.join(LIBRADTRAN_PATH, 'examples/UVSPEC_LOWTRAN_THERMAL.TRANS')),
    'rte_solver': ('disort',),
    'wavelength_grid_file': (os.path.join(LIBRADTRAN_PATH, 'examples/UVSPEC_LOWTRAN_THERMAL.TRANS'),),
    'mol_abs_param': ('LOWTRAN',),
    'wavelength': (3000.0, 5000.0),
    'output_process': ('per_nm',),
    'verbose': ('',)}


HELLO_INPUT = {'wavelength':        (200, 500),
               'rte_solver':        ('disort',),
               'atmosphere_file':   (LIBRADTRAN_PATH + '/data/atmmod/afglus.dat',),
               'source solar':      (LIBRADTRAN_PATH + '/data/solar_flux/atlas_plus_modtran',)}


KATO_FIG3 = {
    'atmosphere_file':  (os.path.join(LIBRADTRAN_PATH, 'examples/AFGLMS50.DAT'),),  # Location of the extraterrestrial spectrum
    'albedo':           (0.8,), #               # Surface albedo
    'sza':              (30.0,) ,                # Solar zenith angle
    'rte_solver':       ('sdisort',),   # Radiative transfer equation solver
    'pseudospherical':  ('',),
    'wavelength':        (3000, 4000),
    'mol_abs_param':    ('KATO',)       # Correlated-k by Kato et al. [1999]
    }    # Calculate integrated solar irradiance


"""
libRadtran Wrapper
~~~~~~~~~~~~~~~~~~
"""


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def write_input(input_definition, pipe, verbose=True, silent=False):
    if not silent:
        print(bcolors.HEADER + '\n**** Input file start ****' + bcolors.ENDC)
    ct = 1
    for key, values in input_definition.iteritems():
        input_line = '{} {}\n'.format(key, ' '.join([str(v) for v in values]))
        pipe.write(input_line)
        if not silent:
            print('{}ln{:02g}:{} {}'.format(bcolors.OKBLUE, ct,bcolors.ENDC, input_line))
        ct += 1
    if not verbose:
        pipe.write('quiet\n')
    if not silent:
        print(bcolors.HEADER + '**** Input file end ****\n\n' + bcolors.ENDC)


def run_radtran(model_definition, silent=False):
    p1 = sp.Popen(UVSPEC_PATH, stdout=sp.PIPE, stdin=sp.PIPE, stderr=sp.PIPE,
                  env={'LIBRADTRAN_DATA_FILES': os.path.join(LIBRADTRAN_PATH, 'data')})
    write_input(model_definition, p1.stdin, silent=silent)
    return p1.communicate()


def parse_output(output, nblocks):
    """

    :type nblocks: int
    """
    all_records = [tuple(float(i) for i in o.split()) for o in output.splitlines()]
    records_out = []
    for block in range(nblocks):
        records_out.append(all_records[block::nblocks])
    return records_out


def read_radtran(output, input_dict):
    try:
        header = input_dict.pop('output_user')
    except:
        if 'disort' in input_dict['rte_solver']:
            header = ['lambda', 'edir', 'edn', 'eup', 'uavgdir', 'uavgdn', 'uavgup']
        else:
            raise Exception('no header for: {}'.format(input_dict['rte_solver']))
    try:
        if 'uu' in header:
            raise Exception()
        umu_list = input_dict.pop('umu')
        try:
            phi_list = input_dict.pop('phi')
            nblocks = 2 + len(phi_list)
        except:
            phi_list = []
            nblocks = 2
    except:
        umu_list = []
        phi_list = []
        nblocks = 1
    records = parse_output(output, nblocks)
    data_frame = pd.DataFrame.from_records(records[0], index='lambda', columns=header)
    if umu_list and not phi_list:
        umu_header = []
        [umu_header.extend(['umu_{}'.format(umu), 'u0u_{}'.format(umu)]) for umu in umu_list]
        umu_frame = pd.DataFrame.from_records(records[1], columns=umu_header, index=data_frame.index)
        data_frame = data_frame.join(umu_frame)
    # can add phi_list after
    return data_frame

def read_radtran_thermal_transmittance(output, input_dict):
    try:
        header = input_dict.pop('output_user')
    except:
        if 'disort' in input_dict['rte_solver']:
            header = ['lambda', 'edir', 'edn', 'eup', 'uavgdir', 'uavgdn', 'uavgup']
        else:
            raise Exception('no header for: {}'.format(input_dict['rte_solver']))
    try:
        if 'uu' in header:
            raise Exception()
        umu_list = input_dict.pop('umu')

    except:
        umu_list = []
        phi_list = []
        nblocks = 1

    records = parse_output(output, nblocks)
    data_frame = pd.DataFrame.from_records(records[0], index='lambda', columns=header)
    if umu_list and not phi_list:
        umu_header = []
        [umu_header.extend(['umu_{}'.format(umu), 'u0u_{}'.format(umu)]) for umu in umu_list]
        umu_frame = pd.DataFrame.from_records(records[1], columns=umu_header, index=data_frame.index)
        data_frame = data_frame.join(umu_frame)
    # can add phi_list after
    return data_frame['uu'].iloc[1]/data_frame['uu'].iloc[0]




def read_transmittance(output, input_):
    records = parse_output(output, 1)

    try:
        data_frame = pd.DataFrame.from_records(records[0], index='lambda', columns=['lambda', 'dir', 'diffuse_up', 'diffuse_down'])
    except:
        data_frame = pd.DataFrame.from_records(records[0], index='lambda', columns=input_['output_user'])
    return data_frame


def read_radtran_basic(output):
    records = parse_output(output)
    return pd.DataFrame.from_records(records, index=0)


UNIT_LU = {'um': -6.0, 'nm': -9.0}

h = 6.62606957e-34
c = 299792458.0
k = 1.3806488e-23
C1 = 1 * h * c ** 2
C2 = (h * c) / k

def L_BB(lambdas, temperature, lambda_unit='um'):
    lambdas *= 10 ** UNIT_LU[lambda_unit]
    A = lambdas ** 5.0 * (np.exp(C2/(lambdas * temperature)) - 1)
    return C1 / A

def write_output(stdout, stderr):
    if stdout:
        print(bcolors.OKBLUE + '**** OUTPUT ****')
        print(stdout)
        print(bcolors.ENDC)
    write_stderr(stderr)

def write_stderr(stderr):
    if stderr:
        print(bcolors.FAIL + '**** ERROR ****')
        print(stderr)
        print(bcolors.ENDC)

def transmittance(input_dict, verbose=False):
    stdout, stderr = run_radtran(input_dict)
    if verbose:
        write_output(stdout, stderr)
    else:
        write_stderr(stderr)
    return read_transmittance(stdout, input_dict)

def radtran_basic(input_dict, verbose=False, silent=False):
    stdout, stderr = run_radtran(input_dict, silent)
    if verbose:
        write_output(stdout, stderr)
    else:
        pass
        #write_stderr(stderr)
    return read_radtran(stdout, input_dict)


def radtran_thermal_transmittance(zout, emiss=1, verbose=False, silent=True):
    albedo = 1 - emiss
    input_dict = {
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'mol_abs_param': ('LOWTRAN',),
         'rte_solver': ('disort',),
         'wavelength_grid_file': ('//home/nsteiner/workspace/flir/out/testh', ),
         'source': ('thermal', '/home/nsteiner/libradtran/examples/UVSPEC_LOWTRAN_THERMAL.TRANS'),
         'sur_temperature': (293, ),

         'umu': (1, ),
         'output_process': ('integrate', ),
         'filter_function_file': ('/home/nsteiner/flir_sr', ),
         'output_user': ('lambda','uu','edir', 'eup', 'edn')}
    input_dict.update({'zout': ('sur', zout), 'albedo': (albedo, )})
    stdout, stderr = run_radtran(input_dict, silent)
    if verbose:
        write_output(stdout, stderr)
    else:
        pass
        #write_stderr(stderr)
    return read_radtran_thermal_transmittance(stdout, input_dict)

def radtran_solar_input(zout, latitude, longitude, datetime_obj, emiss=1, verbose=False, silent=True):
    albedo = 1 - emiss
    input_dict = {
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'mol_abs_param': ('LOWTRAN',),
         'rte_solver': ('disort',),
         'wavelength_grid_file': ('//home/nsteiner/workspace/flir/out/testh', ),
         'source': ('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat', 'per_nm'),
         'sur_temperature': (293, ),
         'umu': (1, ),
         'output_process': ('integrate', ),
         'filter_function_file': ('/home/nsteiner/flir_sr', ),
         'output_user': ('lambda','uu','edir', 'eup', 'edn')}
    input_dict.update({'zout': (zout, ),
                       'albedo': (albedo, ),
                       'time': (datetime_obj.strftime('%Y %m %d %H %M %S'), ),
                       'latitude': ('N', latitude),
                        'longitude': ('W', abs(longitude))})
    stdout, stderr = run_radtran(input_dict, silent)
    if verbose:
        write_output(stdout, stderr)
    else:
        pass
        #write_stderr(stderr)
    return read_radtran(stdout, input_dict)['uu']


def radtran(input_dict=FLIR_SOLAR, pickle_file='solar'):
    stdout, stderr = run_radtran(input_dict)
    write_output(stdout, stderr)
    try:
        read_radtran(stdout, input_dict).to_pickle(pickle_file)
    except:
        try:
            read_radtran_basic(stdout).to_pickle(pickle_file)
        except:
            print('Cannot read output')
    return

BASE_CONFIG = {
     'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
     'mol_abs_param': ('LOWTRAN', ),
     'rte_solver': ('disort', ),
     'zout': (.1, ),
     'umu': (1, ),
     'sza': (60, ),
     'sur_temperature': (293.15, ),
     'phi': (0, ),
     'phi0': (0, ),
     'wavelength_grid_file': ('//home/nsteiner/workspace/flir/out/testh', ),
     'output_process': ('per_nm', ),
     'output_user': ('lambda', 'uu')}

def radtran_full(label='', verbose=False, silent=True, base_config=BASE_CONFIG, **kwargs):
    solar_opt = base_config.copy()
    solar_opt.update(albedo=(1, ),
                     source=('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat', 'per_nm'),
                     **kwargs)
    solar_rad = radtran_basic(solar_opt, verbose=verbose, silent=silent)
    thermal_opt = base_config.copy()
    thermal_opt.update(albedo=(0, ), source=('thermal', ), **kwargs)
    thermal_rad = radtran_basic(thermal_opt, verbose=verbose, silent=silent)
    data_frame = pd.DataFrame(
        {'thermal_'+label: thermal_rad.uu * 1000,
         'solar_'+label: solar_rad.uu,
         'sum_'+label: solar_rad.uu + thermal_rad.uu * 1000}, index=thermal_rad.index)
    return data_frame


def run_sza_sensitivity(steps=5, minmax=(5, 85), surface_temp=293.15):
    range_ = np.linspace(minmax[0], minmax[1], steps)
    df_list = []
    for ct, sza in enumerate(range_):
        print('sza: {}'.format(sza))
        label = '{:02.0f} [deg]'.format(sza)
        df_ = radtran_full(label=label, sza=(sza,), sur_temperature=(surface_temp, ))
        df_list.append(df_)
        if ct == 0:
            N_ = len(df_)
            sum_array = np.empty((N_, len(range_)))
        sum_array[:, ct] = df_['sum_' + label].values
    sum_array_val = (df_.index.values.astype('int32'), range_, sum_array)
    return pd.concat(df_list), sum_array_val


def run_surtemp_sensitivity(steps=5, minmax=(260, 300), sza_deg=60):
    range_ = np.linspace(minmax[0], minmax[1], steps).astype('int32')
    df_list = []
    for ct, surface_temp in enumerate(range_):
        print('surface_temp: {}'.format(surface_temp))
        label = '{:02.0f} $[^oK]$'.format(surface_temp)
        df_ = radtran_full(label=label, sza=(sza_deg,), sur_temperature=(surface_temp, ))
        df_list.append(df_)
        N_ = len(df_)
        if ct == 0:
            sum_array = np.empty((N_, len(range_)))
        sum_array[:, ct] = df_['sum_' + label].values
    sum_array_val = (df_.index.values.astype('int32'), range_, sum_array)
    return pd.concat(df_list), sum_array_val


def integrated_radtran_full(label='', verbose=False, silent=True, **kwargs):
    rad_ = {
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'mol_abs_param': ('LOWTRAN', ),
         'rte_solver': ('disort', ),
         'zout': (.1, ),
         'umu': (1, ),
         'sza': (60, ),
         'phi': (0, ),
         'phi0': (0, ),
         'wavelength_grid_file': ('//home/nsteiner/workspace/flir/out/testh', ),
         'output_process': ('per_nm', ),
         'output_user': ('lambda', 'uu')}
    solar_opt = rad_.copy()
    solar_opt.update(albedo=(1, ),
                     source=('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat', 'per_nm'),
                     **kwargs)
    solar_ = radtran.radtran_basic(solar_opt, verbose=verbose, silent=silent)
    thermal_opt = rad_.copy()
    thermal_opt.update(albedo=(0, ), source=('thermal', ), **kwargs)
    thermal_ = radtran.radtran_basic(thermal_opt, verbose=verbose, silent=silent)
    return pd.DataFrame({'thermal'+label:thermal_.uu * 1000, 'solar'+label:solar_.uu}, index=thermal_.index)



if __name__ == '__main__':

    radtran_thermal_transmittance(.5, .98)




    #sza_all, sza_sum = run_sza_sensitivity(steps=5)
    #sur_all, sur_sum = run_surtemp_sensitivity(steps=5)
    #with open('sens.pck', 'w') as file:
    #    pickle.dump([sza_sum, sur_sum], file)


    '''
    h = (3000.0, 5000.0)
    surf_temp = 330
    solar_source = ('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat')
    solar_solver = 'disort'
    solar_albedo = .8
    thermal_source = ('thermal', '/home/nsteiner/libradtran/examples/UVSPEC_LOWTRAN_THERMAL.TRANS')
    thermal_solver = 'disort'
    solar_ = radtran_basic(
        {'albedo': (solar_albedo,),
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'mol_abs_param': ('LOWTRAN',),
         'rte_solver': (solar_solver,),
         'source': solar_source,
         'sur_temperature': (surf_temp, ),
         'wavelength': h,
         'umu': (1,),
         'zout': (.5, )}, verbose=False)





    tau_0 = transmittance(
        {'albedo': (.9,),
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'latitude': ('N', 66),
         'longitude': ('W', 140),
         'mol_abs_param': ('LOWTRAN',),
         'output_user': ('lambda', 'edir', 'edn', 'eup', 'uu'),
         'rte_solver': ('disort',),
         'source': ('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat'),
         'sur_temperature': (333.15, ),
         'time': (2014, 7, 6, 14, 0, 0),
         'output_quantity': ('transmittance',),
         'wavelength': (2000.0, 6000.0),
         'zout': ('surf',)}, verbose=False)

    tau_1 = transmittance(
        {'albedo': (.8,),
         'atmosphere_file': ('/home/nsteiner/libradtran/data/atmmod/afglus.dat',),
         'latitude': ('N', 66),
         'longitude': ('W', 140),
         'mol_abs_param': ('LOWTRAN',),
         'output_user': ('lambda', 'edir', 'edn', 'eup', 'uu'),
         'rte_solver': ('disort',),
         'source': ('solar', '/home/nsteiner/libradtran/data/solar_flux/kurudz_1.0nm.dat'),
         'sur_temperature': (333.15, ),
         'time': (2014, 7, 6, 14, 0, 0),
         'output_quantity': ('transmittance',),
         'umu': (1,),
         'wavelength': (2000.0, 6000.0),
         'zout': (.2,)}, verbose=False)
    '''

    #lambdas = np.arange(3, 5, .1)
    #l_bb = pd.Series(L_BB(lambdas, 260), index=lambdas)
    #l_bb.to_pickle('surface')
    #radtran_basic(FLIR_SOLAR)
    #main(FLIR_SOLAR, 'solar')
    #main(FLIR_THERMAL, 'thermal')
    #main(FLIR_SOLAR_TOA, 'toa')
    #main(TEST_SOLAR, 'test_solar')
    #main(SOLAR_USS, 'uss_solar')
    #SOLAR_USS_TRANS = SOLAR_USS.copy()
    #SOLAR_USS_TRANS.update({'output_quantity': ('transmittance',),
    #                        'zout': (0, .3, 'toa')})
    #main(SOLAR_USS_TRANS, 'uss_solar_trans_300m')
    #main(SOLAR_USS_TOA, 'uss_solar_toa')
    #main(THERMAL_USS, 'uss_thermal')
    #main(THERMAL_USS_TOA, 'uss_thermal_toa')
    #main(THERMAL_USS_PROFILE, 'uss_thermal_profile')
    #for temp in range(273, 343, 10):
    #    inp = copy.deepcopy(FLIR_THERMAL_TOA)
    #    inp.update({'sur_temperature': (temp, )})
    #    main(inp, './out/thermal_{}'.format(temp))
    #for zout in [.1, .2, .4, .8, 1.6, 3.2, 6.4, 'toa']:
    #    inp = copy.deepcopy(FLIR_THERMAL_TOA)
    #    temp = 303
    #    inp.update({'sur_temperature': (temp, ),
    # #                'zout': (.1, ),
    #                 'albedo': (0,),
    #                 'umu': (np.cos(np.pi/6.0),)})
    #    main(inp, './out/flir_thermal_temp{}_zout{}m_30deg'.format(temp, 0))

    #pass