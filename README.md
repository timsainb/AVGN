
Animal Vocalization Generative Network (AVGN)
==============================

This is a project for taking animal vocalization audio recordings, and learning a generative model of segments of those vocalizations (e.g. syllables) using modern machine learning techniques. Specifically. This package will take in a dataset of wav files, segment them into units (e.g. syllables of birdsong) and train a generative model on those segments. The learned latent representations can be used to cluster syllables in an unsupervised manner, generate novel syllables, visualize sequences, or perform several other analyses.

---


![description](src_img/animalvocalizationfigure.png)


Documentation
------------
Examples of of different songbirds are located in the 'notebooks/birdsong' folder. There is no explicit documentation, but we will work on adding better docstrings to different functions, and adding more notes to the example notebooks. 


Installation
------------

to install run python `setup.py install`

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
