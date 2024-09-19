SNuGGY (Simulation of the Neutrino and Gamma Galactic Yield)
===========================================================

# Requirements:
Python 3 and relevant packages:
- math
- numpy
- scipy
- matplotlib (3.5.0 or higher) (will give error in 3D plot if an earlier version is used)
- astropy

# Installation:
- Download the GitHub repository in your desired directory; given here as:(/path/to/SNuGGY/)
- Add the files to your PYTHONPATH to call the individual python files in the github repository anywhere from your workspace.
``` export PYTHONPATH="/path/to/SNuGGY/:"$PYTHONPATH```

- If you dont want to add the directory to your PYTHONPATH (which is not recomended), you can directly run SNUGGY in the install directory (/path/to/SNuGGY/)

# How to use:
- An example of how to use the package is given in the attached ipython notebook ```run_example_simulation.ipynb```
- Note that ```simulate_nu_g``` is the main code for SNuGGY which calls the other required python files, so calling simulate_nu_g should call the other required python packages too.

# Citing and Acknowledgement:
Please cite the journal article when using this code: [IOP_link](https://iopscience.iop.org/article/10.3847/1538-4357/ad2a5e) 

Thank you.

# Contact
For questions/comments, please contact adesai.physics@gmail.com
