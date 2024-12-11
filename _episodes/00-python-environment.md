---
title: Python Development Environment
teaching: 1
exercises: 1
questions:
- "How to install Python?"
- "How to use conda and pip package managers?"
- "How do you create and activate a virtual environment?"
- "How to install and use Integrated Development Environment (IDE)?"
- "How to use Google Colab?"
objectives:
- "Help participants set up a Python development environment on their machines."
- "Learn about conda and pip package managers."
- "Understand how to create and manage virtual environments."
- "Get hands-on experience on IDEs."
- "Learn how to use Google Colab."
keypoints:
- "Installing Python via Anaconda Distribution."
- "Using conda and pip package managers"
- "Creating a Virtual Environment."
- "Setting Up Integrated Development Environment (IDE)  such as Jupyter Notebook and Visual Studio Code."
- "Using Google Colab"
---

# Setting up the Python Development Environment

## Activities:
- Hands-on installation of Python/Anaconda.
- Setting up Jupyter Notebook.
- Create a virtual environment using conda and pip and install required libraries  (NumPy, Pandas, Matplotlib, scikit-learn, etc.).
- learn how to use Google Colab which provides Jupyter notebook environment with support for Python.

## What is Anaconda?

- Anaconda is a popular **open-source distribution** of Python and R programming languages.
- It simplifies **package management** and **environment management**.
- Designed for **data science**, **machine learning**, **AI**, and **scientific computing**.
- Comes with a **collection of pre-installed libraries** like NumPy, Pandas, Matplotlib, SciPy, Scikit-learn, and more.
- Includes tools like **Jupyter Notebook**, **Spyder**, and **Conda** for streamlined development.

## Importance of Anaconda for Python and Machine Learning Projects

- **Easy Package Management**: Anaconda uses `conda` to install, update, and manage Python packages, avoiding compatibility issues.
- **Environment Isolation**: Allows creation of isolated virtual environments for different projects to prevent library conflicts.
- **Pre-installed Libraries**: Provides a rich ecosystem of pre-installed libraries essential for machine learning and data analysis.
- **Simplified Development**: Tools like Jupyter Notebook make it easy to write and test Python code interactively.
- **Cross-Platform Support**: Available for Windows, macOS, and Linux, ensuring flexibility across different systems.
- **Community and Support**: Widely used in the data science community, making it easier to find resources, tutorials, and support.
- **Optimized for Machine Learning**: Includes popular libraries like TensorFlow, Keras, PyTorch, and Scikit-learn for ML projects.
- **Visualization Tools**: Comes with libraries like Matplotlib, Seaborn, and Plotly for creating insightful visualizations.
- **Scalable**: Suitable for both beginners and advanced users, with capabilities to scale from small projects to complex workflows.

## Install Anaconda on Windows OS

Follow these steps to install Anaconda on your Windows machine:

## Step 1: Download the Installer
- Visit the official [Anaconda Distribution page](https://www.anaconda.com/products/distribution).
- Download the latest Anaconda installer for Windows.

## Step 2: Launch the Installer
- Navigate to your **Downloads** folder.
- Double-click the downloaded installer to start the installation process.

## Step 3: Start the Installation
- Click **Next** on the welcome screen.

## Step 4: Accept the License Agreement
- Read the licensing terms.
- Click **I Agree** to proceed.

> **Tip:** It is recommended to select "Just Me" during installation. This ensures Anaconda is installed only for the current user.

## Step 5: Choose Installation Type
- Click **Next** to confirm your installation type.

## Step 6: Select Installation Folder
- Choose a destination folder for Anaconda.
- Click **Next** to proceed.

> **Important:**  
> - Avoid installing Anaconda in a directory path that contains spaces or Unicode characters.  
> - Do not install as Administrator unless admin privileges are necessary.  
> - **Do not add Anaconda to the PATH environment variable**, as this may cause conflicts with other software.

## Step 7: Install Anaconda
- Click **Install** and wait for the installation to complete.
- Click **Next** to proceed.

## Step 8: Finish Installation
- After a successful installation, you will see the "Thanks for installing Anaconda" dialog box.

## Additional Resources
For more detailed information, refer to the [Anaconda Installation Documentation](https://docs.anaconda.com/anaconda/install/windows/).


## Virtual Environment

A **virtual environment** is an isolated Python environment that allows you to manage dependencies for different projects without conflicts. Below are step-by-step guides for creating virtual environments using **Conda** and **pip**.

### Python Environment Management with Conda

`Conda` is a powerful tool for managing `virtual environments` and `dependencies`.

#### Step 1: Verify Conda and Python Installation

```bash
conda --version
```

```bash
python --version
```
> **Important:**  
> - Ensure Conda is installed. If not, install Anaconda or Miniconda.

### Step 2: Create a New Environment

```bash
conda create --name my_env python=3.9
```
> **Important:**  
> - Replace my_env with your desired environment name and 3.9 with your preferred Python version.

### Step 3: Activate the Environment
```bash
conda activate my_env
```

### Step 4: Install Required Packages
```bash
conda install numpy pandas matplotlib
```

### Step 5: Deactivate the Environment
```bash
conda deactivate
```

### Remove an Environment (Optional)
```bash
conda remove --name my_env --all
```

### Python Environment management with pip

`pip` is Python’s default package manager. Combined with `venv`, it provides a lightweight solution for virtual environments.

### Step 1: Create a Virtual Environment
```bash
python -m venv my_env
```

### Step 2: Activate the Virtual Environment

#### On Windows
```bash
my_env\Scripts\activate
```
#### On macOS/Linux
```bash
source my_env/bin/activate
```
### Step 3: Install Required Packages
```bash
pip install numpy pandas matplotlib
```
### Step 4: Deactivate the Virtual Environment
```bash
deactivate
```
### Step 5: Remove the Virtual Environment (Optional)

#### On macOS/Linux
```bash
rm -r my_env
```
#### On Windows
```bash
rmdir /s my_env
```

## Integrated Development Environment (IDE)

## Google Colab






