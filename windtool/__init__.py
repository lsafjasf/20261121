"""Offline wind-noise annotation, classification and audition tool.

The tool separates three low-frequency phenomena on a two-channel handheld
recorder:

* correlated bass (e.g. kick drum)  -> coherent across microphones,
* independent wind noise            -> incoherent, sustained low-band energy,
* one-sided mic bump/handling       -> strong inter-channel level difference.

Classification combines low-frequency cross-channel coherence, inter-channel
level difference, event persistence and crest factor.  An adaptive,
segment-driven low-band attenuation is then applied, keeping coherent bass
untouched while reducing wind segments.
"""

__version__ = "1.0.0"
