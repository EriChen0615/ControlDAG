# Commandline Design

Typical workflow:

1. Initialize the project `runway init <dest_dir>` -> Initialize directories
2. Create new experiment `runway experiment <exp_name> --copy_from <exp_name>` -> Create experiment folder and config file
3. Prepare data for experiment `runway prepare <exp_name> --copy_from <exp_name>` -> Download/Process data in cache folder
4. Training with experiemnt `runway train <exp_name>` -> produce model and logfiles in experiment folder 
5. Testing with experiment `runway test <exp_name>` -> use model to test, produce decode files and test logs

> Step 3,4,5 can be combined into `runway run <exp_name>`

> Should be able to configure files
