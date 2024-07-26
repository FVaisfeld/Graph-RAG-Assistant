from setuptools import setup, find_packages

setup(
    name='your_project_name',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'openai',
        'neo4j',
        'langchain',
        'ragas'
    ],
    entry_points={
        'console_scripts': [
            'your_script=src.app:main',
        ],
    },
)
