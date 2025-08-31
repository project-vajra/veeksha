Capacity Evaluation
===================

.. toctree::
    :maxdepth: 2
    :hidden:

    capacity_search

``veeksha`` provides a tool through capacity search to find the maximum Queries per Second (QPS) that a given model and inference system can handle given various constraints.

What is Capacity?
-----------------

It is defined as maximum request load (queries-per-second) a system can sustain while meeting certain latency targets (SLOs). Higher capacity reduces the cost of serving requests and improves the user experience.

Steps to Measure Capacity
-------------------------

Capacity Search
~~~~~~~~~~~~~~~
``veeksha`` runs capacity search to find the maximum QPS given certain SLOs.

Refer to :doc:`capacity_search` for more details on how to run capacity search.
