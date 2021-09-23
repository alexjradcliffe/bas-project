# bas-project
A repository for all my files relating to my BAS project on "Probabilistic
radiation belt forecasting".

To read about the methodology of this project, please read report.pdf.

In order to try out this code, please run `process_cdfs.py`, `models.py`, and 
then `plots.py`. This will run them all on the sample directory `example` at 
will produce a series of plots and JSON files in this folder.

It should all be reasonably self-explanatory, and if you want to change the 
directory you're running it on, then change `dir_path` in each of those 
files (it should be immediately under `if __name__ == "__main__"`).

In any directory you want to run it on you must have a config file, which 
should look like the config file in `example/config.json`, but you can use 
it to change any of the parameters for your models. It must also contain a 
folder `cdfs` that holds all the CDF files you want to run it on (these can 
be downloaded from https://rbspgway.jhuapl.edu/psd) and a file 
`Kp_data.lst` containing the Kp data from a timespan that includes all of 
the CDF files (this can be downloaded from
https://omniweb.gsfc.nasa.gov/form/dx1.html by ticking only "Kp*10 Index" 
and clicking "Create file".)

If any of this is unclear (which I suspect some part of it will inevitably 
be), please feel free to get in touch at alexander.j.radcliffe@durham.ac.uk.

Best of luck using it,

Alex Radcliffe
