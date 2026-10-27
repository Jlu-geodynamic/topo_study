# Files used in the study: 
Lithospheric thermal state shapes drainage connectivity and erosion partitioning during continental collision

Yushun Wang, Zhonglan Liu, et al.


Model runs in this study used dealii 9.2.0.

The software version used for this study can be found at: 

ASPECT 2.3

        https://github.com/Djneu/aspect/tree/fault_analysis
	
FastScape commit 18f2588

(elimated)

        https://github.com/fastscape-lem/fastscapelib-fortran 
	
(backup)

        https://github.com/Jlu-geodynamic/fastscape_backup_commit18f2588

All source code used in this study can be found in the backup.


# Additional ASPECT plugins statement:

1."comp" and "rift" were modified from the plugins used in the study: 

Neuharth, D., Brune, S., Wrona, T., Glerum, A., Braun, J., Yuan, X., 2022. Evolution of Rift Systems and Their Fault Networks in Response to Surface Processes. Tectonics 41, e2021TC007166.


We change some initial setting to fit our study.


2."grain" were modified from the sorce code. 


# To run the model and visualize the results:

1.Create a build directory for fastscape and compile it with an added flag for creating a shared library.

        cmake -DBUILD_FASTSCAPELIB_SHARED=ON /path/to/fastscapemake
        make

2.Compile ASPECT with a flag FASTSCAPE_DIR pointing to the fastscape build folder with the shared library

        cmake -DFASTSCAPE_DIR=/path/to/fastscape/build -DDEAL_II_DIR=/path/to/dealii -DCMAKE_BUILD_TYPE=Release /path/to/aspect
        make

3. Using ASPECT to compile plugin. 
   Note: the path of the '#include <files>' in .h and .cc should be modified.

        cmake -DAspect_DIR=/path/to/ASPECT/build ..
        make

4. Change the path of the plugin in the prm files and run.

        mpirun -n [ ] /path/to/aspect/build /path/to/prm/file

5. Use Paraview to visualize results by the .pvsm file.

6. Use Python script to process the topogeaphy results from FastScape.

