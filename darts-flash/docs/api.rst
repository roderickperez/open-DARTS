.. 
   C++ part (interfaced via pybind11)

libflash
########

Global
******

Components
==========
.. autoclass:: dartsflash.libflash.CompData
   :members: 
   :undoc-members:
   :special-members: __init__

Timer
=====
.. autoclass:: dartsflash.libflash.Timer
   :members: 
   :undoc-members:
   :special-members: __init__

Units
=====
.. autoclass:: dartsflash.libflash.Units
   :members: PRESSURE, TEMPERATURE, VOLUME, ENERGY
   :undoc-members:
   :special-members: __init__


Flash methods
*************
.. autoclass:: dartsflash.libflash.FlashParams
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.InitialGuess
   :members: 
   :undoc-members:
   :special-members: __init__

Flash
=====
.. autoclass:: dartsflash.libflash.Flash
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.NegativeFlash2
   :members: 
   :undoc-members:
   :special-members: __init__

Stability
=========
.. autoclass:: dartsflash.libflash.Stability
   :members: 
   :undoc-members:
   :special-members: __init__

PhaseSplit
==========
.. autoclass:: dartsflash.libflash.BaseSplit
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.TwoPhaseSplit
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.MultiPhaseSplit
   :members: 
   :undoc-members:
   :special-members: __init__

Rachford-Rice
=============
.. autoclass:: dartsflash.libflash.RR
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.RR_Eq2
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.RR_EqConvex2
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.RR_EqN
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.RR_Min
   :members: 
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.RR_MinNeg
   :members: 
   :undoc-members:
   :special-members: __init__


Equations of State
******************

EoS
===
.. autoclass:: dartsflash.libflash.EoS
   :members:
   :undoc-members:
   :special-members: __init__

.. .. autoclass:: dartsflash.libflash.IdealGas
..    :members: 
..    :undoc-members:
..    :special-members: __init__

HelmholtzEoS
============
.. autoclass:: dartsflash.libflash.HelmholtzEoS
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.CubicEoS
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.CubicParameters
   :members:
   :undoc-members:
   :special-members: __init__

.. .. autoclass:: dartsflash.libflash.Mix
..    :members:
..    :undoc-members:
..    :special-members: __init__

.. .. autoclass:: dartsflash.libflash.CPA
..    :members: 
..    :undoc-members:
..    :special-members: __init__

.. .. autoclass:: dartsflash.libflash.SAFT
..    :members: 
..    :undoc-members:
..    :special-members: __init__

.. .. autoclass:: dartsflash.libflash.GERG
..    :members: 
..    :undoc-members:
..    :special-members: __init__

AQ
===
.. autoclass:: dartsflash.libflash.AQ
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.AQComposite
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.Ziabakhsh2012
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.Jager2003
   :members:
   :undoc-members:
   :special-members: __init__

VdWP EoS
========
.. autoclass:: dartsflash.libflash.VdWP
   :members:
   :undoc-members:
   :special-members: __init__

.. autoclass:: dartsflash.libflash.Ballard
   :members:
   :undoc-members:
   :special-members: __init__

.. .. autoclass:: dartsflash.libflash.Munck
..    :members:
..    :undoc-members:
..    :special-members: __init__

.. .. autoclass:: dartsflash.libflash.Klauda
..    :members:
..    :undoc-members:
..    :special-members: __init__

PureSolid
=========
.. autoclass:: dartsflash.libflash.PureSolid
   :members:
   :undoc-members:
   :special-members: __init__

.. 
   Python part

dartsflash
##########

EoS Properties
**************

.. automodule:: dartsflash.eos_properties
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Components
**********

.. automodule:: dartsflash.components
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Diagram
*******

.. automodule:: dartsflash.diagram
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
