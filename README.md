<div align="center">  
<h1 align="center">  
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />  
<br>Search-Engine  
</h1>  

  
<p align="center">  
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash" />  
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />  
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />  
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />  
<img src="https://img.shields.io/badge/JSON-000000.svg?style&logo=JSON&logoColor=white" alt="JSON" />  
</p>  
<img src="https://img.shields.io/github/languages/top/elorberb/Search-Engine?style&color=5D6D7E" alt="GitHub top language" />  
<img src="https://img.shields.io/github/languages/code-size/elorberb/Search-Engine?style&color=5D6D7E" alt="GitHub code size in bytes" />  
<img src="https://img.shields.io/github/commit-activity/m/elorberb/Search-Engine?style&color=5D6D7E" alt="GitHub commit activity" />  
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="GitHub license" />  
</div>  
  
---  
  
## 📖 Table of Contents  
- [📖 Table of Contents](#-table-of-contents)  
- [📍 Overview](#-overview)  
- [📂 Repository Structure](#-repository-structure)  
- [⚙️ Modules](#%EF%B8%8F-modules)  
- [🚀 Getting Started](#-getting-started)  
    - [🔧 Installation](#-installation)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [👏 Acknowledgments](#-acknowledgments)  
 
  
---  
  
## 📍 Overview  
    
This is a university project aims to create a search engine that operates on Wikipedia data, leveraging techniques like cosine similarity and tf-idf, as well as utilizing pre-trained models like word2vec. The project is divided into several modules, each one with a unique responsibility that contributes to the overall functionality.  
  
---  
  
## 📂 Repository Structure  
  
```sh
└── Search-Engine/
    ├── README.md
    ├── extra_files/
    │   ├── queries_train.json
    │   ├── run_frontend_in_colab.ipynb
    │   ├── run_frontend_in_gcp.sh
    │   └── startup_script_gcp.sh
    └── src/
        ├── __init__.py
        ├── backend.py
        ├── bm25.py
        ├── bucket_manipulation.py
        ├── core_functions.py
        ├── inverted_index_gcp.py
        ├── metrics.py
        ├── retrievel_functions.py
        ├── search_frontend.py
        ├── search_functions.py
        └── tests/
            ├── __init__.py
            ├── testing_engine.ipynb
            └── tests.py
```

## ⚙️ Modules  
  
<details closed><summary>Src</summary>  
  
| File                                                                                                     | Summary                   |  
| ---                                                                                                      | ---                       |  
| [backend.py](https://github.com/elorberb/Search-Engine/blob/main/src/backend.py)                         | Code for downloading the indexes and other relevant files from our BUCKETS |  
| [bm25.py](https://github.com/elorberb/Search-Engine/blob/main/src/bm25.py)                               | Module for building the bm25 search function |  
| [core_functions.py](https://github.com/elorberb/Search-Engine/blob/main/src/core_functions.py)           | General functions for use in the modules of the rest of the project, such as tokenizing, mapping to title, and more |  
| [bucket_manipulation.py](https://github.com/elorberb/Search-Engine/blob/main/src/bucket_manipulation.py) | A module whose purpose is to connect to Buckets and download the indexes and other files from them in order to perform the retrieval |  
| [metrics.py](https://github.com/elorberb/Search-Engine/blob/main/src/metrics.py)                         | A module that contains all the evaluation functions we built in order to test the search functions |  
| [retrievel_functions.py](https://github.com/elorberb/Search-Engine/blob/main/src/retrievel_functions.py) | All the functions that are intended for calculating tf-idf for queries and documents and calculating cosine similarity in order to build the search body |  
| [search_functions.py](https://github.com/elorberb/Search-Engine/blob/main/src/search_functions.py)       | All the search functions we built in order to retrieve documents according to the various indexes and methods we used |  
| [search_frontend.py](https://github.com/elorberb/Search-Engine/blob/main/src/search_frontend.py)         | The module given to us in order to do a test against a local server |  
| [inverted_index_gcp.py](https://github.com/elorberb/Search-Engine/blob/main/src/inverted_index_gcp.py)   | The code for building the skeleton of the inverted index object |  
  
</details>  


<details closed><summary>Extra_files</summary>  
  
| File                                                                                                                       | Summary                   |  
| ---                                                                                                                        | ---                       |  
| [startup_script_gcp.sh](https://github.com/elorberb/Search-Engine/blob/main/extra_files/startup_script_gcp.sh)             | Script for starting up the GCP |  
| [run_frontend_in_gcp.sh](https://github.com/elorberb/Search-Engine/blob/main/extra_files/run_frontend_in_gcp.sh)           | Script for running the frontend in GCP |  
| [run_frontend_in_colab.ipynb](https://github.com/elorberb/Search-Engine/blob/main/extra_files/run_frontend_in_colab.ipynb) | Notebook for running the frontend in Google Colab |  
  
</details>  


---

## 🚀 Getting Started

### 🔧 Installation

#### Clone the Search-Engine repository:
```sh
git clone https://github.com/elorberb/Search-Engine
```

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License  
  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for additional info.  
  
---  
  
## 👏 Acknowledgments  
  
- Oren Fix, my work partner  
- The university that provided us with this project 
