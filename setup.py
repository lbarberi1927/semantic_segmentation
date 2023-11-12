from setuptools import setup, find_packages

setup(
    name="san",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask",
        "werkzeug",
        "gradio",
    ],
    entry_points={
        "console_scripts": [
            "san_server=scripts.san_server:main",
        ],
    },
)
