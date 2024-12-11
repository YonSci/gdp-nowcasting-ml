---
title: Virtual Environments
teaching: 1
exercises: 1
questions:
- "What is a virtual environment?"
- "How do virtual environments help manage dependencies?"
- "How do you create and activate a virtual environment?"
objectives:
- "Learn about virtual environments and their purpose in Python projects."
- "Understand how to create and manage virtual environments."
- "Get hands-on experience with setting up and using a virtual environment."
keypoints:
- "Virtual environments isolate dependencies."
- "Using `venv` for creating environments."
- "Activating and deactivating virtual environments."
---

# Python Environment Management

Managing Python environments is essential for keeping project dependencies isolated and avoiding version conflicts. Python provides several tools for managing environments, including **Conda** and the built-in **venv** module. Both help to create isolated environments for Python projects, each with its own set of dependencies.

## Conda Environments

Conda is a package, dependency, and environment management tool that simplifies managing different Python versions and libraries. It is especially popular in the scientific community and on Windows platforms.

### What is a Conda Environment?

A Conda environment is a self-contained directory that contains a specific collection of Conda packages. These environments help isolate different projects from each other, ensuring that dependencies for one project do not interfere with others.

> Avoid installing packages into the base Conda environment. Create a new environment for each project to maintain isolation.

### Creating Environments with Conda (Linux/MacOS)

To create a new Conda environment, use the following command:

~~~
admin@MacBook~ $ conda create -n <env_name> python=<version#>
~~~  
{: .bash}

For example, to create a Conda environment named `ml-climate` with Python 3.12:

~~~
admin@MacBook~ $ conda create --name ml-climate python=3.12
~~~  
{: .bash}

To activate the environment:

~~~
admin@MacBook~ $ conda activate ml-climate
~~~  
{: .bash}

### Installing Packages in Conda Environments

Install packages within an environment using `conda` or `pip`:

~~~
admin@MacBook~ $ conda install <package_name>
~~~  
{: .bash}

Or, if the package is not available via Conda:

~~~
admin@MacBook~ $ pip install <package_name>
~~~  
{: .bash}

### Exporting and Sharing Environments

Export an environment configuration to a file for sharing with colleagues:

~~~
admin@MacBook~ $ conda env export --no-builds --file environment.yaml
~~~  
{: .bash}

This creates a `YAML` file that lists all packages and dependencies used in the environment. Others can use this file to recreate the same environment.

### Deactivating and Removing Environments

To deactivate a Conda environment:

~~~
admin@MacBook~ $ conda deactivate
~~~  
{: .bash}

To remove an environment:

~~~
admin@MacBook~ $ conda env remove --name <env_name>
~~~  
{: .bash}

## Virtual Environments with `venv`

The `venv` module, included in Python by default, is another tool for creating isolated environments. While Conda is a more comprehensive solution, `venv` is lightweight and works well for basic Python projects.

### Creating a Virtual Environment with `venv`

To create a virtual environment, follow these steps:

1. **Open a Terminal**:
   - On Linux/MacOS, press `Ctrl + Alt + T` or search for `Terminal`.

2. **Navigate to the Desired Directory**:
   Use `cd` to move to the project folder:

   ~~~
   admin@MacBook~ $ cd Documents/ml-climate
   ~~~  
   {: .bash}

3. **Create the Virtual Environment**:
   Use the following command:

   ~~~
   admin@MacBook~ $ python3 -m venv ml-climate
   ~~~  
   {: .bash}

4. **Activate the Virtual Environment**:
   - On Linux/MacOS:

   ~~~
   admin@MacBook~ $ source ml-climate/bin/activate
   ~~~  
   {: .bash}

5. **Install Packages**:
   Install dependencies with `pip`:

   ~~~
   admin@MacBook~ $ pip install <package_name>
   ~~~  
   {: .bash}

6. **Deactivate the Virtual Environment** (Optional):
   When done, deactivate the environment:

   ~~~
   admin@MacBook~ $ deactivate
   ~~~  
   {: .bash}

### Managing Packages with `venv`

While in the virtual environment, use `pip` to install, update, or uninstall packages:

- Install a package:

  ~~~
  admin@MacBook~ $ pip install <package_name>
  ~~~  
  {: .bash}

- To list installed packages:

  ~~~
  admin@MacBook~ $ pip list
  ~~~  
  {: .bash}

- To uninstall a package:

  ~~~
  admin@MacBook~ $ pip uninstall <package_name>
  ~~~  
  {: .bash}

---

## Exercise: Create and Use a Virtual Environment

1. **Create a New Virtual Environment**:
   - Name it `myproject` and install `requests`.

2. **Write a Python Script**:
   - Create a file `fetch_page.py` using `requests` to fetch a webpage.

   Example script:

   ~~~python
   import requests

   response = requests.get('https://www.example.com')
   print(response.text)
   ~~~  
   {: .python}

3. **Activate the Virtual Environment** and install `requests`:

   ~~~
   admin@MacBook~ $ source myproject/bin/activate
   ~~~  
   {: .bash}

   ~~~
   admin@MacBook~ $ pip install requests
   ~~~  
   {: .bash}

4. **Run the Script** and deactivate the environment when done:

   ~~~
   admin@MacBook~ $ python fetch_page.py
   ~~~  
   {: .bash}

   ~~~
   admin@MacBook~ $ deactivate
   ~~~  
   {: .bash}


## Creating Conda Environments on Windows (PowerShell)

To set up Python environments using **Conda** on Windows via **PowerShell**, follow these steps. Ensure that **Anaconda** or **Miniconda** is installed before proceeding.

### Installing Conda

1. **Download and Install Conda**:
   - Download the Anaconda or Miniconda installer for Windows from their official websites.
   - Run the installer and follow the setup instructions, making sure to check the option to add Conda to your system’s PATH.

2. **Verify Conda Installation**:
   Open **PowerShell** and check if Conda is installed:

   ~~~
   PS C:\> conda --version
   ~~~  
   {: .bash}

   If Conda is installed correctly, it should return the version number.

### Creating and Managing Conda Environments

Once Conda is set up, you can create and manage environments as follows:

1. **Create a New Environment**:
   To create a new environment with a specific Python version, use the `conda create` command. For example, to create an environment called `ml-climate` with Python 3.12:

   ~~~
   PS C:\> conda create --name ml-climate python=3.12
   ~~~  
   {: .bash}

2. **Activate the Environment**:
   After creating the environment, activate it using the `conda activate` command:

   ~~~
   PS C:\> conda activate ml-climate
   ~~~  
   {: .bash}

3. **Install Packages**:
   Once the environment is activated, you can install packages like this:

   ~~~
   PS C:\> conda install <package_name>
   ~~~  
   {: .bash}

4. **Deactivating the Environment**:
   To exit the environment and return to the base environment, run:

   ~~~
   PS C:\> conda deactivate
   ~~~  
   {: .bash}

5. **Remove an Environment**:
   If you no longer need an environment, you can remove it with the following command:

   ~~~
   PS C:\> conda env remove --name <env_name>
   ~~~  
   {: .bash}

### Exporting and Sharing Environments on Windows

To export the configuration of a Conda environment to a `.yaml` file (useful for sharing), run:

~~~
PS C:\> conda env export --no-builds --file environment.yaml
~~~  
{: .bash}

This creates a `YAML` file that lists all installed packages, dependencies, and channels used for installation.

### Installing Packages from a YAML File

To recreate an environment from a `.yaml` file, use the following command:

~~~
PS C:\> conda env create --file environment.yaml
~~~  
{: .bash}



Here’s the revised exercise with the environment name changed to `ml-flow`.

---

## Exercise: Conda Environment Setup on Windows

### Objective:
- Learn how to create, activate, and manage Conda environments on Windows using **PowerShell**.
- Practice installing packages and exporting the environment configuration.

### Tasks:

1. **Install Conda**:
   - If you haven't already, download and install **Anaconda** or **Miniconda** from their official websites.
   - Verify the installation by running this command in PowerShell:
     ~~~
     PS C:\> conda --version
     ~~~
     You should see the version of Conda installed.

2. **Create a New Conda Environment**:
   - Create a Conda environment named `ml-flow` with Python 3.10:
     ~~~
     PS C:\> conda create --name ml-flow python=3.10
     ~~~
   - Confirm the environment creation when prompted.

3. **Activate the New Environment**:
   - Activate the `ml-flow` environment:
     ~~~
     PS C:\> conda activate ml-flow
     ~~~
   - You should see `(ml-flow)` before the command prompt, indicating that the environment is active.

4. **Install Packages**:
   - Install two packages: `numpy` and `matplotlib`:
     ~~~
     PS C:\> conda install numpy matplotlib
     ~~~
   - Wait for the installation to complete.

5. **Verify Package Installation**:
   - Check if `numpy` and `matplotlib` are installed by running:
     ~~~
     PS C:\> conda list
     ~~~
   - You should see `numpy` and `matplotlib` in the list of installed packages.

6. **Deactivate the Environment**:
   - Deactivate the environment and return to the base environment:
     ~~~
     PS C:\> conda deactivate
     ~~~

7. **Export the Environment Configuration**:
   - Export the `ml-flow` environment configuration to a YAML file:
     ~~~
     PS C:\> conda env export --no-builds --file ml-flow.yaml
     ~~~
   - Check that the `ml-flow.yaml` file has been created in your current directory.

8. **Remove the Environment**:
   - Finally, remove the environment after you're done:
     ~~~
     PS C:\> conda env remove --name ml-flow
     ~~~
   - Confirm that the environment is removed by running `conda env list`.



