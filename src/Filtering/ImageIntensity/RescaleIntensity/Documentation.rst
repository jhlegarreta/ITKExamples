:name: RescaleIntensity

Rescale an Image
================

.. index::
   single: RescaleIntensityImageFilter

Synopsis
--------


Rescale a grayscale image


Results
-------

.. figure:: Gourds.png
  :scale: 50%
  :alt: Input image

  Input image

.. figure:: OutputBaseline.png
  :scale: 50%
  :alt: Output image

  Output image


Code
----

Python
......

.. literalinclude:: Code.py
   :language: python
   :lines: 1, 16-

C++
...

.. literalinclude:: Code.cxx
   :lines: 18-


Classes demonstrated
--------------------

.. breathelink:: itk::RescaleIntensityImageFilter
