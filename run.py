#from examples import gto_error, estimate, sensitivity, adapt
from examples import plot, adapt
import matplotlib as mpl

# Setting matplotlib params here
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams["pdf.use14corefonts"] = True
mpl.rcParams['lines.markersize'] = 7.5
mpl.rcParams['lines.markerfacecolor'] = 'none'
mpl.rcParams["legend.edgecolor"] = 'k'
mpl.rcParams["legend.labelspacing"] = 0.01
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams["axes.labelpad"] = 1.4
mpl.rcParams['axes.linewidth'] = 0.7


# Run all the simulations
# -----------------------

basis="cc-pvdz unc-cc-pvdz unc-cc-pvtz pc-1 unc-pc-1 pc-2 unc-pc-2 \
    cc-pvtz aug-cc-pvdz aug-cc-pvtz aug-cc-pvqz cc-pvqz cc-pv5z \
    aug-cc-pv5z pc-3 pc-4 aug-pc-3 aug-pc-4 unc-pc-4"

# split expression etc

# Run simulations from Section 4.2.3.
#gto_error.main()
#estimate.main()
plot.main()

# Run simulations from Section 4.2.4
#sensitivity.main()

# Run simulations from Section 4.2.5
adapt.main()



