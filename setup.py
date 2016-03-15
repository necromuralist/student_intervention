try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

with open('readme.rst') as reader:
    long_description = reader.read()
    
setup(name='student_intervention',
      long_description=long_description,
      version= '2016.03.12',
      description="Student Intervention System.",
      author="russell",
      platforms=['linux'],
      url = '',
      author_email="necromuralist@gmail.com",
      license = "MIT",
      install_requires = 'numpy pandas matplotlib seaborn sklearn'.split(),
      packages = find_packages(),
      )

# an example last line would be cpm= cpm.main: main

# If you want to require other packages add (to setup parameters):
# install_requires = [<package>],
#version=datetime.today().strftime("%Y.%m.%d"),
# if you have an egg somewhere other than PyPi that needs to be installed as a dependency, point to a page where you can download it:
# dependency_links = ["http://<url>"]
