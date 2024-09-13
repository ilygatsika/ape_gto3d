#from examples import gto_error, estimate, sensitivity, adapt
from examples import plot, adapt
import matplotlib as mpl
import scienceplots

# Setting matplotlib params
# User can customize this globally
# -----------------------

mpl.rcParams["legend.labelspacing"] = 0.01
mpl.pyplot.style.use('science')
mpl.rcParams["legend.fancybox"] = True


# Run all the simulations
# -----------------------

basis="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"

# split expression etc

# Run simulations from Section 4.2.3.
#gto_error.main()
plot.main()

# TODO test and pass two arguments: basis and grid as
# not correct for the moment
# >> python3 run.py "basis" "grid"
#estimate.main()

# Run simulations from Section 4.2.4
#sensitivity.main()

# Run simulations from Section 4.2.5
adapt.main()



