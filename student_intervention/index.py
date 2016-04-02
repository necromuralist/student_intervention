
# python standard library
import os
from distutils.util import strtobool

# this package
from commoncode.index_builder import create_toctree

PDF = strtobool(os.environ.get('PDF', 'no'))

if not PDF:
    output = "A pdf version is available :download:`here <student_intervention_submission/student_intervention.pdf>` and the repository for the source of this document is `here <https://github.com/necromuralist/student_intervention>`_."
else:
   output = "The repository for the source of this document is `here <https://github.com/necromuralist/student_intervention>`_."
print(output)