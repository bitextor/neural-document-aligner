#!/usr/bin/env python3

import setuptools
import os

def reqs_from_file(src):
    requirements = []

    with open(src) as f:
        for line in f:
            line = line.strip()

            if line.startswith("-r"):
                add_src = line.split(' ')[1]
                add_req = reqs_from_file(add_src)

                requirements.extend(add_req)
            elif line.startswith("-"):
                # Ignore the rest of flags (e.g. -U)

                requirements.append(line.split(" ", 1)[-1])
            else:
                requirements.append(line)

    return requirements

if __name__ == "__main__":
    wd = os.path.dirname(os.path.abspath(__file__))
    requirements = reqs_from_file(f"{wd}/requirements.txt")

    with open(f"{wd}/README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="neural-document-aligner",
        version="1.0",
        install_requires=requirements,
        license="GNU General Public License v3.0",
        #author=,
        #author_email=,
        #maintainer=,
        #maintainer_email,
        description="Document aligner which uses neural technologies to search matches across bilingual documents",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/bitextor/neural-document-aligner",
        packages=["neural_document_aligner", "neural_document_aligner.utils"],
        #classifiers=[],
        #project_urls={},
        package_data={
            "": ["scripts/*",
            ]
        },
        entry_points={
            "console_scripts": [
                "neural-document-aligner = neural_document_aligner.neural_document_aligner:main_wrapper",
            ]
        }
        )
