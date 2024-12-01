# COMP 8085 Project 2

## Installation

1. Open a terminal in the project root directory.
2. Run `py -m venv ./p2`.
3. Set `./p2/Scripts/python.exe` as the python interpreter in your dev environment.
4. Run `p2/Scripts/activate`.
5. Run `python -m pip install -r requirements.txt`.
6. If supported, run `python -m pip install torch --index-url https://download.pytorch.org/whl/cu118`
6. Else, run `python -m pip install torch`
7. Run `deactivate`.

Linux instructions:
1. Make sure `python3-pip` and `virtualenv` are installed.
2. CD to project root.
3. Run `virtualenv -p python3 p2`.
4. Run `source p2/bin/activate`.
5. Run `sudo pip3 install -r requirements.txt`.
6. Run `deactivate`.

## Using the Project

#### Start

1. Open terminal in the project root directory.
2. Run `p2/Scripts/activate` or `p2\Scripts\activate` to activate the virtual environment.
3. Run `python -m pip install -r requirements.txt` to make sure your local venv is up-to-date.

Linux instructions:
1. Open terminal in the project root directory.
2. Run `source p2/bin/activate` to activate the virtual environment.
3. Run `pip3 install -r requirements.txt` to make sure your local venv is up-to-date.

#### Execution

1. Make sure the above commands have been executed.
<!--2. Run `python3 src/NIDS.py <args>`-->
\* Note that different operating systems may use different slash separators/

#### Close

1. Open terminal in the project root directory.
2. Run `pip freeze --exclude torch > requirements.txt` to add any possible new requirements.
3. Run `deactivate` to exit the virtual environment.

Linux instructions:
1. Open terminal in the project root directory.
2. Run `pip3 freeze > requirements.txt` to add any possible new requirements.
3. Run `deactivate` to exit the virtual environment.

## Environment Structure

- `/datasets` contains the prepared datasets for use in tweaking, training, testing, and validation.
This is local and should not be committed to the repository.
- `/p2` contains the virtual environment files. This is local and should not be committed to the repository.
- `/src` contains the python source files for the project.
- `/models` contains serialized binaries of the preset model weights.
