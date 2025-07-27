"""
This is the small script of changing the parameters of Input__* files.

How to use it:
  1. Set up the files and the parameters you want ot change.
    a. The file is set to be a `File()` class. The `File` takes four inputs: file name, constant
       parameters, changing parameters, and flag file or not. Please check out the `Classes` section
       for the detail of inputs.
    b. Wrap all the `File` classes to a list called `files`.

    * NOTICE:
      Parameter files:
        The first column specifies the option's name and the second column should be the value of the option.
      Flag files:
        The first line(row) of the flag files should always start with `# ` and the following column
        names in the header specify the options' names.

  2. Overwrite the `execution()` function to your execution.
     For instance, `subprocess.called(['./gamer']).

  3. You might also want to pass some arguments to the `execution()` from the command line. We already
     pass all the arguments, so all you need to do is to add you own `parser.add_argument` at `Main`
     section.

An example of the script (from Alexander Kunkel):
-----------------------------------------------------------------------------------------------------
# Additional imports for interacting with file system
import shutil
import glob

...

def execution( **kwargs ):
    cwd               = os.getcwd()
    record_parameters = kwargs["record"]
    exe               = kwargs["exe"]


    # Run GAMER with executable passed as command line parameter
    subprocess.run([kwargs["exe"]], capture_output=False)

    # Analyse run: Call python scripts etc.
    os.system("python3 plot_wave_slice.py -s 1 -e 1")
    os.system("python3 analyse_l1_error.py")
    os.system("ls -alt > Record__FileSizes")

    # Create folder recording which parameters where changed
    par_dir  = exe[2:] + "_" + "_".join(record_parameters.keys())

    try:
        os.mkdir(par_dir)
    except FileExistsError:
        pass

    # Create folder recording current runtime parameters
    sub_dir  = str(record_parameters)
    dest_dir = os.path.join(par_dir, sub_dir)
    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        pass

    # Move and or output files to folder, delete with os.remove() (files) and shutil.rmtree() (directories) if necessary
    for file in glob.glob(r'*.png'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Input*'):
        shutil.copy(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Record*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'VortexPairLinear*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Data*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    return

...

file_name1       = "Input__Parameter"
loop_paras1      = { "OPT__FIXUP_RESTRICT": [0, 1], "OPT__REF_FLU_INT_SCHEME": [4, 5, 6, 7],
const_paras1     = {"MAX_LEVEL": 1, "NX0_TOT_X": 64, "NX0_TOT_Y": 64, "OPT__INIT": 2, "END_STEP": 1000, "REFINE_NLEVEL": 1, "REGRID_COUNT":1, "OPT__INT_PHASE": 0}

file_name2       = "Input__Flag_Rho"
loop_paras2      = { "Density": [0, 1e3]}
const_paras2     = {}


file1  = File( file_name1, const_paras1, loop_paras1, False  )
file2  = File( file_name2, const_paras2, loop_paras2, True  )

files  = [ file1, file2 ]
-----------------------------------------------------------------------------------------------------

For developer:
1. The main concept is to use a recursion function instead of nest for loops, so the code stays clean
   and easy to maintain.
2. First we iterate the files then the parameters of each files. At the end of iteration, we called
   `execution()`.
"""
#====================================================================================================
# Import packages
#====================================================================================================
import numpy as np
import argparse
import re
import os
import sys
import subprocess



#====================================================================================================
# Global variables
#====================================================================================================
RECURSION_LIMIT = 1000  # Reset the recursion limit. (default in python is 1000)
RETURN_FAIL    = False
RETURN_SUCCESS = True



#====================================================================================================
# Classes
#====================================================================================================
class File():
    def __init__( self, file_name, consts, paras, flag ):
        """
        file_name : string. The file name.
        consts    : dict. The const parameter want to be specified.
        paras     : dict with list element. The parameter want to be chenged.
        flag      : bool. The flag file or not.
        """
        self.name   = file_name
        self.paras  = paras
        self.consts = consts
        self.flag   = flag


#====================================================================================================
# Functions
#====================================================================================================
def iter_files( files, record_changed={}, **kwargs ):
    # copy_files = files.copy() # work in python3 only
    copy_files = list(files)

    if copy_files == []:        # End of changing files
        return execution( record=record_changed, **kwargs )

    replace_file_class = copy_files.pop(0)
    replace_file  = replace_file_class.name
    replace_paras = replace_file_class.paras
    replace_flag  = replace_file_class.flag
    return iter_paras( replace_file, replace_paras, record_changed, replace_flag, rest_files=copy_files, **kwargs )

def iter_paras( file_name, paras, record, flag, **kwargs ):
    copy_paras  = paras.copy()
    copy_kwargs = kwargs.copy()
    copy_kwargs.pop("rest_files")
    if copy_paras == {}:        # End of the changes
        return iter_files( kwargs["rest_files"], **copy_kwargs )

    replace_key = next(iter(copy_paras)) # Take the first key to replace
    vals = copy_paras.pop(replace_key)
    for val in vals:
        if not kwargs["quite"]: print("File <%s> changing %s --> %s"%(file_name, replace_key, str(val)))
        replace_parameter( file_name, replace_key, val, flag )
        record[replace_key]=val
        iter_paras( file_name, copy_paras, record, flag, **kwargs )  # repalce the next parameter
    return


def replace_parameter( file_name, name, val, flag ):
    if flag:
        with open( file_name, 'r' ) as f:
            header = f.readline()
        index = { v:i for i, v in enumerate(header.split()[1:]) }       # The first element assume to be "#"
        data = np.loadtxt(file_name)
        if len(index) != data.shape[1]: raise BaseException("ERROR: The number of columns in <%s> does not match to the header."%(file_name))
        data[:, index[name]] = val
        np.savetxt(file_name, data, fmt="%7g"+"%20.16g"*(len(index)-1), header=header[2:-1])
    else:
        with open( file_name, 'r' ) as f:
            content = f.read()

        content_new, n_sub = re.subn( r"^(%s\s+)([^\s]+)"%name, r"\g<1>%s"%str(val), content, flags=re.M )
        if n_sub == 0: raise BaseException("ERROR: Can not find <%s> in <%s>."%(name, file_name))

        with open( file_name, 'w' ) as f:
            f.write( content_new )
    return


# Additional imports for interacting with file system
import shutil
import glob
import subprocess

def execution( **kwargs ):
    cwd               = os.getcwd()
    record_parameters = kwargs["record"]
    exe               = kwargs["exe"]


    # Run GAMER with executable passed as command line parameter
    subprocess.run([kwargs["exe"]], capture_output=False)

    # Analyse run: Call python scripts etc.
    #os.system("python3 plot_wave_slice.py -s 1 -e 1")
    #os.system("python3 analyse_l1_error.py")
    #os.system("ls -alt > Record__FileSizes")

    # Create folder recording which parameters where changed
    par_dir  = "results/" + exe[2:] + "_" + "_".join(record_parameters.keys())

    try:
        os.mkdir(par_dir)
    except FileExistsError:
        pass

    # Create folder recording current runtime parameters
    sub_dir  = str(record_parameters)
    dest_dir = os.path.join(par_dir, sub_dir)
    try:
        os.mkdir(dest_dir)
    except FileExistsError:
        pass

    # Move and or output files to folder, delete with os.remove() (files) and shutil.rmtree() (directories) if necessary
    for file in glob.glob(r'*.png'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Input*'):
        shutil.copy(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Record*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'PlaneWave*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    for file in glob.glob(r'Data*'):
        shutil.move(os.path.join(cwd, file), os.path.join(cwd, dest_dir))

    return


#====================================================================================================
# Main
#====================================================================================================
sys.setrecursionlimit(RECURSION_LIMIT)                                          # Reset the recursion depth limit

# 1. Set up the files and the parameters to be changed
file_name1   = "Input__Parameter"                                               # file name
const_paras1 = {}                                      # the parameters change only once
loop_paras1  = {}                     # the parameters change as given list
# set the `File` class, the file changing method is decided by the last parameter. If the file is a flag file, please set it to `True`.
file1        = File( file_name1, const_paras1, loop_paras1, flag=False )

file_name2   = "Input__TestProb"
const_paras2 = {}
loop_paras2  = {"PWave_NWavelength":2*np.pi/np.array([0.011,
 0.015822948771164292,
 0.02276051889226269,
 0.032739865857944514,
 0.047094656385913355,
 0.06774330321726293,
 0.09744534694510915,
 0.14017024842734474,
 0.20162787819156802,
 0.2900315988603394,
 0.4171959209805475,
 0.6001154259285372,
 0.8632359673866076,
 1.2417216808531581,
 1.7861544131075937,
 2.569293615999135,
 3.0,
 3.695800114912161,
 5.316223262428928,
 5.5,
 6, 
 7.647120757953166,
 11.0])/4}#np.array([1, 2, 3, 4, 6, 9, 13, 19, 27, 40, 57, 83, 119, 172, 247, 355, 512])}
file2        = File( file_name2, const_paras2, loop_paras2, flag=False )


# Wrap all the `File` classes.
files = [file1, file2]

# 2. Taking the input arguments
parser = argparse.ArgumentParser( description = "A small script of changing the parameters of Input__* files.",
                                  formatter_class = argparse.RawTextHelpFormatter,
                                  add_help=False)

parser.add_argument( "-h", "--help",
                     action="help", default=argparse.SUPPRESS,
                     help="Show this help message and exit.\n"
                   )

parser.add_argument( "-q", "--quite",
                     action="store_true",
                     help="Enable slient mode.\n"
                   )

parser.add_argument( "-e", "--exe", type=str,
                     default="./fd_4_conserve_mass",
                     help="The file you want to execute.\n"
                   )

args = vars( parser.parse_args() )

# 3. Set the constant parameters
for f in files:
    for key, val in f.consts.items():
        replace_parameter( f.name, key, val, f.flag )

# 4. Start changing parameters and running
iter_files( files, **args )
