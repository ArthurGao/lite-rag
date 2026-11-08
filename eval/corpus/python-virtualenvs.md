# Python virtual environments

## Why use one

A virtual environment keeps a project's packages separate from the system Python and from other projects. Two projects can then depend on different versions of the same library without breaking each other.

## Creating an environment

Run `python -m venv .venv` in the project folder. This creates a `.venv` directory holding its own interpreter and its own site-packages.

## Activating it

On macOS and Linux run `source .venv/bin/activate`; on Windows run `.venv\Scripts\activate`. The shell prompt then shows the environment's name, and `python` and `pip` refer to the copies inside it. Run `deactivate` to leave.

## Recording dependencies

`pip freeze > requirements.txt` writes the exact installed versions to a file. Someone else can recreate the same set with `pip install -r requirements.txt`.

## What not to commit

Never commit the `.venv` folder itself to version control; it is large and specific to one machine. Commit the requirements file instead and recreate the environment from it.
